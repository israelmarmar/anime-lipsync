"""
character_detect.py
===================
Detecção de personagem por texto usando Florence-2.

Fluxo por frame:
  1. Florence2 com task <CAPTION_TO_PHRASE_GROUNDING> localiza o personagem
     descrito pelo texto no frame completo → bbox do personagem
  2. O bbox é expandido com character_margin e salvo em character_bboxes/
  3. O crop do personagem é passado para o YOLO de face (phase0)
  4. As coordenadas da face são remapeadas de volta para o frame completo

Integração com phase0:
  - Se character_query não for vazio, `locate_character()` é chamado antes
    do YOLO de face, restringindo a busca ao crop do personagem.
  - Se Florence2 não detectar o personagem em um frame, usa o bbox do frame
    anterior (ou frame inteiro como fallback).

Dependências:
  pip install transformers timm einops  (Florence-2)
  O modelo é carregado uma vez e mantido em memória durante o pipeline.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


# ===========================================================================
# Loader singleton (carrega Florence-2 uma única vez por processo)
# ===========================================================================

_florence_lock  = threading.Lock()
_florence_model = None
_florence_proc  = None
_florence_device = "cpu"


def _load_florence(model_id: str = "microsoft/Florence-2-base") -> None:
    """Carrega Florence-2 na memória (thread-safe, idempotente)."""
    global _florence_model, _florence_proc, _florence_device

    with _florence_lock:
        if _florence_model is not None:
            return
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "pip install transformers timm einops  # necessário para Florence-2"
            ) from e

        _florence_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Florence2] Carregando {model_id} em {_florence_device}...")

        _florence_proc  = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True)
        _florence_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if _florence_device == "cuda" else torch.float32,
            trust_remote_code=True,
            attn_implementation="eager",   # evita erro _supports_sdpa em transformers recentes
        ).to(_florence_device).eval()

        print("[Florence2] Modelo pronto.")


def _unload_florence() -> None:
    """Descarrega Florence-2 da memória para liberar VRAM após a Fase 0."""
    global _florence_model, _florence_proc

    with _florence_lock:
        if _florence_model is not None:
            del _florence_model
            _florence_model = None
        if _florence_proc is not None:
            del _florence_proc
            _florence_proc = None

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Florence2] Modelo descarregado.")


# ===========================================================================
# Detecção de personagem por texto
# ===========================================================================

def locate_character(
    img_np: np.ndarray,
    query: str,
    margin: float = 0.05,
    florence_model_id: str = "microsoft/Florence-2-base",
) -> Optional[Tuple[int, int, int, int]]:
    """
    Localiza o personagem descrito por `query` no frame usando Florence-2
    com a task CAPTION_TO_PHRASE_GROUNDING.

    Parâmetros
    ----------
    img_np  : frame RGB uint8
    query   : texto descrevendo o personagem (ex: "the girl with blue hair")
    margin  : expansão proporcional do bbox detectado (default 5%)

    Retorna
    -------
    (x1, y1, x2, y2) em coordenadas absolutas do frame, ou None se não detectado.
    """
    _load_florence(florence_model_id)

    H, W = img_np.shape[:2]
    pil  = Image.fromarray(img_np)

    task   = "<CAPTION_TO_PHRASE_GROUNDING>"
    prompt = f"{task} {query}"

    with _florence_lock:
        dtype  = next(_florence_model.parameters()).dtype
        inputs = _florence_proc(
            text=prompt, images=pil, return_tensors="pt"
        ).to(_florence_device)
        # garante que pixel_values tem o mesmo dtype do modelo (float16 em CUDA)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            ids = _florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=1,
                do_sample=False,
                use_cache=False,   # fix: evita AttributeError em past_key_values com transformers recentes
            )

        result = _florence_proc.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = _florence_proc.post_process_generation(
            result,
            task=task,
            image_size=(W, H),
        )

    bboxes = parsed.get(task, {}).get("bboxes", [])
    labels = parsed.get(task, {}).get("labels", [])

    if not bboxes:
        return None

    # Usa o primeiro bbox (Florence ordena por relevância)
    x1, y1, x2, y2 = [int(v) for v in bboxes[0]]
    label = labels[0] if labels else "?"
    print(f"    [Florence2] '{label}' → ({x1},{y1},{x2},{y2})")

    # Aplica margem
    mx = int((x2 - x1) * margin)
    my = int((y2 - y1) * margin)
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(W, x2 + mx); y2 = min(H, y2 + my)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


# ===========================================================================
# Cache de bbox por frame (propagação temporal)
# ===========================================================================

class CharacterTracker:
    """
    Mantém o último bbox conhecido do personagem para propagação temporal:
    se Florence2 falhar em um frame, usa o bbox do frame anterior.

    Thread-safe — cada frame worker tem sua própria instância local,
    mas o estado inicial (frame 0) é compartilhado via `seed_bbox`.
    """

    def __init__(self, seed_bbox: Optional[Tuple[int, int, int, int]] = None):
        self._last = seed_bbox
        self._lock = threading.Lock()

    def update(self, bbox: Optional[Tuple[int, int, int, int]]) -> None:
        if bbox is not None:
            with self._lock:
                self._last = bbox

    def get(self) -> Optional[Tuple[int, int, int, int]]:
        with self._lock:
            return self._last


# ===========================================================================
# Detecção sequencial para inicializar o tracker (frame 0 → N, single-thread)
# ===========================================================================

def build_character_bboxes(
    store,                        # DiskStore
    n_frames: int,
    query: str,
    margin: float = 0.05,
    florence_model_id: str = "microsoft/Florence-2-base",
    char_bbox_dir: Optional[Path] = None,
) -> dict:
    """
    Roda Florence-2 em todos os frames sequencialmente e salva os bboxes.
    Retorna dict {frame_idx: (x1,y1,x2,y2)}.

    Roda ANTES da fase paralela de preprocessing para evitar contenção
    na GPU com os workers YOLO.
    """
    import json

    print(f"[Florence2] Detectando '{query}' em {n_frames} frames...")
    _load_florence(florence_model_id)

    bboxes: dict = {}
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    log_step = max(1, n_frames // 10)

    for i in range(n_frames):
        img_np = store.load_orig(i)
        bbox   = locate_character(img_np, query, margin, florence_model_id)

        if bbox is None and last_bbox is not None:
            # Propaga o bbox do frame anterior
            bbox = last_bbox
            print(f"    [Florence2] frame {i}: não detectado → propagando bbox anterior")
        elif bbox is None:
            print(f"    [Florence2] frame {i}: não detectado e sem bbox anterior")

        bboxes[i] = bbox
        last_bbox  = bbox

        if char_bbox_dir is not None and bbox is not None:
            char_bbox_dir.mkdir(parents=True, exist_ok=True)
            p = char_bbox_dir / f"{i:06d}.json"
            p.write_text(json.dumps({"x1": bbox[0], "y1": bbox[1],
                                     "x2": bbox[2], "y2": bbox[3]}))

        if (i + 1) % log_step == 0 or i == n_frames - 1:
            print(f"  [Florence2] {i+1}/{n_frames} frames processados")

    _unload_florence()
    detected = sum(1 for v in bboxes.values() if v is not None)
    print(f"[Florence2] Concluído: {detected}/{n_frames} frames com bbox.")
    return bboxes