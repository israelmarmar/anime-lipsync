import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .constants import MOUTH_TYPE_NAMES
from .store import DiskStore
from .utils import cuda_cleanup, vram_info
from .phase0 import _dynamic_upscale

try:
    from rembg import remove as _rembg_remove
    _REMBG_AVAILABLE = True
except ImportError:
    _REMBG_AVAILABLE = False


def _detect_mouth(face_np: np.ndarray,
                  face_meta: dict,
                  face_bbox,
                  nomouth_shape: Tuple[int, int],
                  mouth_model,
                  upscale_crop_face: float,
                  conf_mouth: float,
                  mouth_type: int = 0,
                  mouth_padding_per_type: Dict[int, int] = None,
                  mouth_brightness_per_type: Dict[int, float] = None) -> Optional[Tuple[np.ndarray, int, int, int, int]]:
    """
    Detecta a boca na face gerada, faz rembg e devolve
    (rgba, abs_x, abs_y, abs_w, abs_h) ou None.

    Para neutral_closed (mouth_type=0): o YOLO localiza a boca, mas o recorte
    é feito como retângulo sólido (sem segmentação por máscara). O rembg é
    aplicado em seguida para remover a pele do retângulo — evitando que a
    boca fechada seja cortada pela metade pelo segmentador.

    Para half_open e fully_open (mouth_type=1,2): comportamento original —
    o YOLO segmenta e o rembg refina o alpha.
    """
    mouth_padding    = (mouth_padding_per_type    or {}).get(mouth_type, 0)
    mouth_brightness = (mouth_brightness_per_type or {}).get(mouth_type, 1.0)

    fH, fW = face_np.shape[:2]
    if fH == 0 or fW == 0:
        return None

    H_fr, W_fr = nomouth_shape
    fx1, fy1, fx2, fy2 = face_bbox if face_bbox else (0, 0, fW, fH)

    crop_w_orig = int(face_meta.get("crop_w", 0)) or (fx2 - fx1)
    crop_h_orig = int(face_meta.get("crop_h", 0)) or (fy2 - fy1)

    # Garante 1:1 com o frame (defensivo)
    if (fW != crop_w_orig or fH != crop_h_orig) and crop_w_orig > 0 and crop_h_orig > 0:
        face_np = cv2.resize(face_np, (crop_w_orig, crop_h_orig),
                             interpolation=cv2.INTER_LANCZOS4)
        fH, fW = face_np.shape[:2]

    # Upscale para YOLO detectar com maior precisão
    up_W, up_H = _dynamic_upscale(fH, fW, upscale_crop_face)
    face_up = cv2.resize(face_np, (up_W, up_H), interpolation=cv2.INTER_LANCZOS4)

    with torch.no_grad():
        results = mouth_model(face_up, verbose=False, conf=conf_mouth)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        del results, boxes, face_up
        return None

    best = int(np.argmax(boxes.conf.cpu().numpy()))
    bx1u, by1u, bx2u, by2u = boxes.xyxy.cpu().numpy()[best].astype(int)
    bx1u = max(0, bx1u); by1u = max(0, by1u)
    bx2u = min(up_W, bx2u); by2u = min(up_H, by2u)
    del results, boxes, face_up

    if bx2u <= bx1u or by2u <= by1u:
        return None

    # Remap upscale → face_np (1:1 com frame)
    sx = fW / up_W; sy = fH / up_H
    bx1 = max(0,  int(bx1u * sx)); by1 = max(0,  int(by1u * sy))
    bx2 = min(fW, int(bx2u * sx)); by2 = min(fH, int(by2u * sy))

    if bx2 <= bx1 or by2 <= by1:
        return None

    # Aplica padding ao bbox da boca (expande antes de recortar)
    if mouth_padding > 0:
        pad = mouth_padding
        bx1 = max(0, bx1 - pad); by1 = max(0, by1 - pad)
        bx2 = min(fW, bx2 + pad); by2 = min(fH, by2 + pad)

    mouth_crop = face_np[by1:by2, bx1:bx2].copy()

    # Ajuste de brilho do crop da boca
    if abs(mouth_brightness - 1.0) > 0.001:
        mouth_crop = np.clip(
            mouth_crop.astype(np.float32) * mouth_brightness, 0, 255
        ).astype(np.uint8)

    if mouth_type == 0:
        # ── neutral_closed: máscara híbrida (bordas + elipse) ────────────────
        # Problema com rembg: a cor dos lábios fechados é muito próxima da pele,
        # então o rembg apaga as linhas junto com o fundo.
        # Problema com elipse pura: o feather suaviza as bordas do elipse, mas
        # as linhas dos lábios ficam exatamente nessa região de transição — alpha
        # baixo demais para aparecer na composição final.
        #
        # Nova abordagem em 3 camadas:
        #   1. "edge_mask": Canny no canal L (luminância) detecta as linhas
        #      escuras dos lábios e as dilata para ter espessura visível.
        #   2. "region_mask": elipse LARGA (95%×85%) com feather MÍNIMO —
        #      serve apenas para excluir os cantos do bbox, não cortar os lábios.
        #   3. alpha final = max(edge_mask, region_mask) — garante que qualquer
        #      pixel que seja borda de lábio OU dentro do elipse seja preservado.
        ch, cw = mouth_crop.shape[:2]

        # --- Camada 1: bordas de lábio via Canny ---
        gray = cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2GRAY)
        # Equalização leve para realçar contraste dos lábios em relação à pele
        gray_eq = cv2.equalizeHist(gray)
        # Canny com thresholds adaptados ao tamanho do recorte
        lo = max(20, int(cw * 0.5))
        hi = max(60, int(cw * 1.5))
        edges = cv2.Canny(gray_eq, lo, hi)
        # Dilata as bordas detectadas para cobrir a linha de lábio com espessura
        dil_r = max(2, min(ch, cw) // 8)
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_r * 2 + 1, dil_r * 2 + 1))
        edge_mask = cv2.dilate(edges, dil_k, iterations=1).astype(np.float32) / 255.0
        # Suavização leve para anti-aliasing nas bordas
        edge_ksize = max(3, (dil_r | 1))
        edge_mask = cv2.GaussianBlur(edge_mask, (edge_ksize, edge_ksize), 0)

        # --- Camada 2: elipse larga com feather mínimo (só exclui cantos) ---
        region = np.zeros((ch, cw), dtype=np.uint8)
        cx_e, cy_e = cw // 2, ch // 2
        # Elipse bem generosa: 95% da largura, 85% da altura
        axes = (max(1, int(cw * 0.475)), max(1, int(ch * 0.425)))
        cv2.ellipse(region, (cx_e, cy_e), axes, 0, 0, 360, 255, -1)
        # Feather mínimo — só suaviza a borda do elipse, não encolhe o interior
        feather_k = max(3, (min(ch, cw) // 12) | 1)
        region_mask = cv2.GaussianBlur(region, (feather_k, feather_k), 0).astype(np.float32) / 255.0

        # --- Camada 3: combina — preserva bordas de lábio E interior do elipse ---
        mask_f = np.maximum(edge_mask, region_mask)
        # Garante que a máscara não ultrapasse 1.0 e não introduza ruído fora da elipse
        # (pixels fora do elipse expandido ficam com alpha apenas da borda, se houver)
        outer_ellipse = np.zeros((ch, cw), dtype=np.uint8)
        axes_outer = (max(1, int(cw * 0.50)), max(1, int(ch * 0.50)))
        cv2.ellipse(outer_ellipse, (cx_e, cy_e), axes_outer, 0, 0, 360, 255, -1)
        outer_f = cv2.GaussianBlur(outer_ellipse, (feather_k, feather_k), 0).astype(np.float32) / 255.0
        # Fora do elipse externo, mantém apenas as bordas detectadas (evita artefatos de pele)
        mask_f = np.where(outer_f > 0.05, mask_f, edge_mask * outer_f * 4)
        mask_f = np.clip(mask_f, 0.0, 1.0)

        # Pré-multiplica RGB pelo alpha para composite() desmultiplicar corretamente
        rgb_pm = np.clip(mouth_crop.astype(np.float32) * mask_f[:, :, None], 0, 255).astype(np.uint8)
        alpha8 = (mask_f * 255).clip(0, 255).astype(np.uint8)
        mouth_rgba = np.dstack([rgb_pm, alpha8])
    else:
        # ── half_open / fully_open: rembg (segmentação por cor funciona bem) ─
        mouth_rgba = np.array(
            _rembg_remove(Image.fromarray(mouth_crop)).convert("RGBA"), dtype=np.uint8)

    del mouth_crop

    # Remap 1:1 — face_np já está em crop_w_orig × crop_h_orig
    abs_x = max(0, min(W_fr - 1, fx1 + bx1))
    abs_y = max(0, min(H_fr - 1, fy1 + by1))
    abs_w = min(max(1, bx2 - bx1), W_fr - abs_x)
    abs_h = min(max(1, by2 - by1), H_fr - abs_y)

    return mouth_rgba, abs_x, abs_y, abs_w, abs_h


def process_single(q: int,
                   store: DiskStore,
                   mouth_types: List[int],
                   mouth_model,
                   bbox_cache: Dict,
                   upscale_crop_face: float,
                   conf_mouth: float,
                   mouth_padding_per_type: Dict[int, int] = None,
                   mouth_brightness_per_type: Dict[int, float] = None) -> None:
    mt        = mouth_types[q]
    meta      = store.load_face_frame_meta(q)
    fg        = int(meta.get("face_group", 0))
    cache_key = (mt, fg)

    if cache_key in bbox_cache:
        store.save_mouth_frame(q, *bbox_cache[cache_key])
        return

    if not store.has_face_frame(q):
        store.save_mouth_frame(q, np.zeros((4, 4, 4), dtype=np.uint8), 0, 0, 0, 0)
        return

    face_np       = store.load_face_frame(q)
    face_meta     = store.load_face_frame_meta(q)
    face_bbox     = store.load_face_bbox(q)
    nomouth_np    = store.load_nomouth(q)
    nomouth_shape = nomouth_np.shape[:2]
    del nomouth_np

    fH, fW = face_np.shape[:2]
    if fH == 0 or fW == 0:
        store.save_mouth_frame(q, np.zeros((4, 4, 4), dtype=np.uint8), 0, 0, 0, 0)
        return

    result = _detect_mouth(face_np, face_meta, face_bbox, nomouth_shape,
                           mouth_model, upscale_crop_face, conf_mouth,
                           mouth_type=mt,
                           mouth_padding_per_type=mouth_padding_per_type,
                           mouth_brightness_per_type=mouth_brightness_per_type)

    if result is None:
        store.save_mouth_frame(q, np.zeros((4, 4, 4), dtype=np.uint8), 0, 0, 0, 0)
        return

    bbox_cache[cache_key] = result
    store.save_mouth_frame(q, *result)
    print(f"  [F2] CANONICAL {MOUTH_TYPE_NAMES[mt]} fg={fg} repr=frame{q} "
          f"→ ({result[1]},{result[2]}) {result[3]}×{result[4]}px")
    cuda_cleanup()


def _worker(q: int, store: DiskStore, mouth_types: List[int],
            mouth_model_path: str, bbox_cache: Dict,
            bbox_cache_lock: threading.Lock,
            upscale_crop_face: float, conf_mouth: float,
            mouth_padding_per_type: Dict[int, int] = None,
            mouth_brightness_per_type: Dict[int, float] = None) -> int:
    from ultralytics import YOLO
    try:
        model = YOLO(mouth_model_path)
        local: Dict = {}
        process_single(q, store, mouth_types, model, local,
                       upscale_crop_face, conf_mouth,
                       mouth_padding_per_type, mouth_brightness_per_type)
        del model
        with bbox_cache_lock:
            for k, v in local.items():
                if k not in bbox_cache:
                    bbox_cache[k] = v
        cuda_cleanup()
        return q
    except Exception as exc:
        print(f"[F2-Worker] Erro frame {q}: {exc}")
        try:
            store.save_mouth_frame(q, np.zeros((4, 4, 4), dtype=np.uint8), 0, 0, 0, 0)
        except Exception:
            pass
        return -1


def run_phase2(store: DiskStore,
               mouth_types: List[int],
               n_frames: int,
               mouth_model_path: str,
               n_workers: int,
               upscale_crop_face: float = 2.0,
               conf_mouth: float = 0.1,
               mouth_padding_per_type: Dict[int, int] = None,
               mouth_brightness_per_type: Dict[int, float] = None) -> None:
    if not _REMBG_AVAILABLE:
        raise ImportError("pip install rembg")

    print(f"\n[F2] Bocas paralela — {n_frames} frames, {n_workers} workers, "
          f"upscale=auto(min={upscale_crop_face:.1f}x) conf={conf_mouth} "
          f"padding={mouth_padding_per_type} brightness={mouth_brightness_per_type}...")

    # Etapa 2a: um frame representativo por chave canônica
    repr_frames: Dict[Tuple[int, int], int] = {}
    for q in range(n_frames):
        try:
            meta = store.load_face_frame_meta(q)
            fg   = int(meta.get("face_group", 0))
            key  = (mouth_types[q], fg)
            if key not in repr_frames:
                repr_frames[key] = q
        except Exception:
            pass

    bbox_cache: Dict = {}
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_worker, q, store, mouth_types, mouth_model_path,
                      bbox_cache, lock, upscale_crop_face, conf_mouth,
                      mouth_padding_per_type, mouth_brightness_per_type): q
            for q in repr_frames.values()
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[F2a] EXCEÇÃO: {e}")

    print(f"[F2] {len(bbox_cache)} bboxes canônicas detectadas.")

    # Etapa 2b: propaga para todos os frames
    for q in range(n_frames):
        if store.has_mouth_frame(q):
            continue
        try:
            meta = store.load_face_frame_meta(q)
            fg   = int(meta.get("face_group", 0))
            key  = (mouth_types[q], fg)
            with lock:
                canonical = bbox_cache.get(key)
            if canonical:
                store.save_mouth_frame(q, *canonical)
            else:
                store.save_mouth_frame(q, np.zeros((4, 4, 4), dtype=np.uint8), 0, 0, 0, 0)
        except Exception as e:
            print(f"[F2b] Frame {q}: {e}")
            try:
                store.save_mouth_frame(q, np.zeros((4, 4, 4), dtype=np.uint8), 0, 0, 0, 0)
            except Exception:
                pass

    cuda_cleanup()
    print(f"[F2] Concluída. {vram_info()}")
