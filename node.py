"""
LipSync Z-Turbo Pipeline V5.5
Custom Node ComfyUI — orquestrador principal.

Módulos:
  constants.py  — constantes, mapeamentos de fonema/boca
  utils.py      — VRAM, tensor, áudio, JSON, ffmpeg, máscaras
  store.py      — DiskStore, VramGuard
  phonemes.py   — AllosaurusDetector
  lora.py       — apply_lora
  phase0.py     — YOLO unico face+boca + inpainting
  phase1.py     — Diffusion Z-Turbo + LoRA por tipo
  phase2.py     — Detecção de boca + rembg
  phase3.py     — Composição + overlap workers
  node.py       — este arquivo (registro do nó)
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from queue import Queue
from typing import Any, Dict

import torch

from .constants import MOUTH_TYPE_NAMES
from .phonemes import AllosaurusDetector
from .store import DiskStore
from .utils import (
    comfy_audio_to_wav, save_audio_original,
    get_comfy_output_dir, unique_filename,
    build_video_with_audio, compute_workers,
    t2np, cuda_cleanup,
)
from .phase0 import run_phase0
from .character_detect import build_character_bboxes
from .phase1 import run_phase1
from .phase2 import run_phase2, smooth_mouth_tracks
from .phase3 import run_phase3, launch_overlap_workers, ensure_all_outputs


class LipSyncZTurboPipeline:
    """LipSync Z-Turbo Pipeline V5.5"""

    CATEGORY     = "LipSync"
    FUNCTION     = "run"
    OUTPUT_NODE  = True
    RETURN_TYPES = ("STRING", "VHS_VIDEOINFO", "STRING")
    RETURN_NAMES = ("video_path", "video_info", "debug_dir")

    @classmethod
    def INPUT_TYPES(cls):
        node_dir = os.path.dirname(os.path.abspath(__file__))
        external_detection_model = "/disco3/anime_mouth_yolo/best.pt"
        detection_model_default = (
            external_detection_model
            if os.path.exists(external_detection_model)
            else os.path.join(node_dir, "best.pt")
        )
        return {
            "required": {
                "audio":            ("AUDIO",),
                "images":           ("IMAGE",),
                "model":            ("MODEL",),
                "model_patch":      ("MODEL_PATCH",),
                "clip":             ("CLIP",),
                "vae":              ("VAE",),
                "detection_model_path": ("STRING", {
                    "default": detection_model_default,
                    "multiline": False,
                    "tooltip": "Modelo YOLO unico: classe 0 = face, classe 1 = boca.",
                }),
            },
            "optional": {
                # ── Geral ────────────────────────────────────────────────────
                "fps":                   ("INT",     {"default": 12,   "min": 1,    "max": 60}),
                "source_fps":            ("FLOAT",   {"default": 12.0, "min": 0.1,  "max": 120.0}),
                "lang_id":               ("STRING",  {"default": "uni", "multiline": False}),
                "sim_threshold":         ("FLOAT",   {"default": 0.92, "min": 0.0,  "max": 1.0,   "step": 0.01}),
                "motion_variance_factor": ("FLOAT",  {
                    "default": 0.001, "min": 0.0, "max": 0.2, "step": 0.001,
                    "tooltip": "Limiar relativo de movimento do bbox da face para reiniciar caches de boca. Menor = mais sensivel."
                }),
                "mask_dilation":         ("INT",     {"default": 8,    "min": 0,    "max": 64}),
                "mask_blur":             ("INT",     {"default": 5,    "min": 0,    "max": 31}),
                "vram_safety_margin_mb": ("INT",     {"default": 1024, "min": 256,  "max": 16384, "step": 256}),
                "compose_feather_px":    ("INT",     {"default": 2,    "min": 0,    "max": 32}),
                "enable_overlap":        ("BOOLEAN", {"default": True}),
                "use_rembg":             ("BOOLEAN", {"default": True, "tooltip": "Ativa BEN2 no recorte da boca para remover pele residual. Mais preciso, porém mais lento."}),
                "upscale_crop_face":     ("FLOAT",   {"default": 2.0,  "min": 1.0,  "max": 10.0,  "step": 0.5}),
                "mouth_conf":            ("FLOAT",   {"default": 0.1,  "min": 0.01, "max": 1.0,   "step": 0.01}),
                "remove_mouth":          ("BOOLEAN", {"default": True}),
                "save_debug_folder":     ("BOOLEAN", {"default": False, "tooltip": "Copia a pasta temporária de trabalho para o output do ComfyUI e retorna o caminho em debug_dir."}),
                "verify_generated_mouth": ("BOOLEAN", {"default": True, "tooltip": "Valida com YOLO se a face gerada contém boca antes de cachear/sobrepor."}),
                "mouth_regen_attempts":  ("INT",     {"default": 2,    "min": 0,    "max": 8, "tooltip": "Quantas novas amostras KSampler tentar quando a boca gerada não for detectada."}),
                "use_open_for_half":     ("BOOLEAN", {"default": False, "tooltip": "Gera fully_open para half_open e reduz apenas a altura do recorte na composição."}),
                "half_open_height_scale": ("FLOAT", {"default": 0.55, "min": 0.1, "max": 1.0, "step": 0.05, "tooltip": "Escala vertical da boca aberta usada como half_open."}),
                "hed_detector_mode":      (["auto", "controlnet_aux", "comfy_hed"], {
                    "default": "auto",
                    "tooltip": (
                        "Fonte do mapa HED/scribble usado no ControlNet. "
                        "controlnet_aux usa HEDdetector(scribble); auto cai para HEDPreprocessor; "
                        "comfy_hed força o node HEDPreprocessor do ComfyUI."
                    ),
                }),
                # ── Padding de boca por tipo ────────────────────────────────────
                "mouth_padding_closed": ("INT",   {"default": 0,   "min": 0,   "max": 64, "tooltip": "Padding (px) do recorte da boca neutral_closed antes do rembg/composicao."}),
                "mouth_padding_half":   ("INT",   {"default": 0,   "min": 0,   "max": 64, "tooltip": "Padding (px) do recorte da boca half_open antes do rembg/composicao."}),
                "mouth_padding_open":   ("INT",   {"default": 0,   "min": 0,   "max": 64, "tooltip": "Padding (px) do recorte da boca fully_open antes do rembg/composicao."}),
                # ── Brilho de boca por tipo ──────────────────────────────────────
                "mouth_brightness_closed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": "Brilho do recorte da boca neutral_closed. >1 clareia, <1 escurece."}),
                "mouth_brightness_half":   ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": "Brilho do recorte da boca half_open. >1 clareia, <1 escurece."}),
                "mouth_brightness_open":   ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05, "tooltip": "Brilho do recorte da boca fully_open. >1 clareia, <1 escurece."}),

                "video_output_filename": ("STRING",  {"default": "lipsync_output", "multiline": False}),
                # ── Personagem (Florence-2) ──────────────────────────────────
                "enable_character_detect": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Ativa a detecção de personagem via Florence-2. "
                        "Quando desativado, o pipeline usa a imagem inteira para localizar a face "
                        "e os campos character_query / character_margin / florence_model_id são ignorados."
                    ),
                }),
                "character_query": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Texto descrevendo o personagem a sincronizar (ex: 'the girl with blue hair'). "
                        "Só tem efeito quando enable_character_detect = True E este campo não estiver vazio. "
                        "Florence-2 localiza o personagem antes do YOLO de face."
                    ),
                }),
                "character_margin": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Expansão proporcional do bbox do personagem detectado pela Florence-2.",
                }),
                "florence_model_id": ("STRING", {
                    "default": "microsoft/Florence-2-base",
                    "multiline": False,
                    "tooltip": "ID do modelo Florence-2 no HuggingFace (ex: microsoft/Florence-2-large).",
                }),
                # ── LoRA — neutral_closed ────────────────────────────────────
                "lora_closed_path":           ("STRING", {"default": "closed_mouth_z_image_turbo_lora.safetensors", "multiline": False}),
                "lora_closed_strength_model": ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "lora_closed_strength_clip":  ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                # ── LoRA — half_open ─────────────────────────────────────────
                "lora_half_path":           ("STRING", {"default": "half_open_z_image_turbo_lora.safetensors", "multiline": False}),
                "lora_half_strength_model": ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "lora_half_strength_clip":  ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                # ── LoRA — fully_open ────────────────────────────────────────
                "lora_open_path":           ("STRING", {"default": "full_open_z_image_turbo_lora.safetensors", "multiline": False}),
                "lora_open_strength_model": ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "lora_open_strength_clip":  ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                # ── KSampler — neutral_closed ────────────────────────────────
                "closed_steps":       ("INT",    {"default": 4,    "min": 1, "max": 30}),
                "closed_denoise":     ("FLOAT",  {"default": 0.60, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "closed_cfg":         ("FLOAT",  {"default": 1.2,  "min": 0.0, "max": 10.0, "step": 0.1}),
                "closed_cn_strength": ("FLOAT",  {"default": 0.60, "min": 0.0, "max": 2.0,  "step": 0.05}),
                "closed_sampler":     ("STRING", {"default": "res_multistep", "multiline": False}),
                "closed_scheduler":   ("STRING", {"default": "simple",        "multiline": False}),
                # ── KSampler — half_open ─────────────────────────────────────
                "half_steps":         ("INT",    {"default": 4,    "min": 1, "max": 30}),
                "half_denoise":       ("FLOAT",  {"default": 0.80, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "half_cfg":           ("FLOAT",  {"default": 1.0,  "min": 0.0, "max": 10.0, "step": 0.1}),
                "half_cn_strength":   ("FLOAT",  {"default": 0.60, "min": 0.0, "max": 2.0,  "step": 0.05}),
                "half_sampler":       ("STRING", {"default": "res_multistep", "multiline": False}),
                "half_scheduler":     ("STRING", {"default": "simple",        "multiline": False}),
                # ── KSampler — fully_open ────────────────────────────────────
                "open_steps":         ("INT",    {"default": 4,    "min": 1, "max": 30}),
                "open_denoise":       ("FLOAT",  {"default": 0.90, "min": 0.0, "max": 1.0,  "step": 0.01}),
                "open_cfg":           ("FLOAT",  {"default": 1.2,  "min": 0.0, "max": 10.0, "step": 0.1}),
                "open_cn_strength":   ("FLOAT",  {"default": 0.60, "min": 0.0, "max": 2.0,  "step": 0.05}),
                "open_sampler":       ("STRING", {"default": "res_multistep", "multiline": False}),
                "open_scheduler":     ("STRING", {"default": "simple",        "multiline": False}),
                # ── CONDITIONING externos ────────────────────────────────────
                "cond_closed": ("CONDITIONING",),
                "cond_half":   ("CONDITIONING",),
                "cond_open":   ("CONDITIONING",),
            },
        }

    def run(
        self,
        audio, images, model, model_patch, clip, vae,
        detection_model_path: str,
        # Geral
        fps: int = 12, source_fps: float = 12.0,
        lang_id: str = "uni", sim_threshold: float = 0.92,
        motion_variance_factor: float = 0.001,
        mask_dilation: int = 8, mask_blur: int = 5,
        vram_safety_margin_mb: int = 1024,
        compose_feather_px: int = 2,
        enable_overlap: bool = True,
        use_rembg: bool = True,
        upscale_crop_face: float = 2.0,
        mouth_conf: float = 0.1,
        remove_mouth: bool = True,
        save_debug_folder: bool = False,
        verify_generated_mouth: bool = True,
        mouth_regen_attempts: int = 2,
        use_open_for_half: bool = False,
        half_open_height_scale: float = 0.55,
        hed_detector_mode: str = "auto",
        # Padding por tipo
        mouth_padding_closed: int = 0,
        mouth_padding_half: int = 0,
        mouth_padding_open: int = 0,
        # Brilho por tipo
        mouth_brightness_closed: float = 1.0,
        mouth_brightness_half: float = 1.0,
        mouth_brightness_open: float = 1.0,

        video_output_filename: str = "lipsync_output",
        # Personagem
        enable_character_detect: bool = False,
        character_query: str = "",
        character_margin: float = 0.05,
        florence_model_id: str = "microsoft/Florence-2-base",
        # LoRA
        lora_closed_path: str = "closed_mouth_z_image_turbo_lora.safetensors",
        lora_closed_strength_model: float = 1.0,
        lora_closed_strength_clip:  float = 1.0,
        lora_half_path:   str = "half_open_z_image_turbo_lora.safetensors",
        lora_half_strength_model:   float = 1.0,
        lora_half_strength_clip:    float = 1.0,
        lora_open_path:   str = "full_open_z_image_turbo_lora.safetensors",
        lora_open_strength_model:   float = 1.0,
        lora_open_strength_clip:    float = 1.0,
        # KSampler
        closed_steps=4, closed_denoise=0.60, closed_cfg=1.2,
        closed_cn_strength=0.60, closed_sampler="res_multistep", closed_scheduler="simple",
        half_steps=4,   half_denoise=0.80,   half_cfg=1.0,
        half_cn_strength=0.60,   half_sampler="res_multistep",   half_scheduler="simple",
        open_steps=4,   open_denoise=0.90,   open_cfg=1.2,
        open_cn_strength=0.60,   open_sampler="res_multistep",   open_scheduler="simple",
        cond_closed=None, cond_half=None, cond_open=None,
    ) -> dict:

        t_start = time.perf_counter()

        mouth_padding_per_type = {
            0: mouth_padding_closed,
            1: mouth_padding_half,
            2: mouth_padding_open,
        }
        mouth_brightness_per_type = {
            0: mouth_brightness_closed,
            1: mouth_brightness_half,
            2: mouth_brightness_open,
        }

        cfg_per_type = {
            0: dict(steps=closed_steps, denoise=closed_denoise, cfg=closed_cfg,
                    controlnet_strength=closed_cn_strength,
                    sampler=closed_sampler, scheduler=closed_scheduler),
            1: dict(steps=half_steps,   denoise=half_denoise,   cfg=half_cfg,
                    controlnet_strength=half_cn_strength,
                    sampler=half_sampler,   scheduler=half_scheduler),
            2: dict(steps=open_steps,   denoise=open_denoise,   cfg=open_cfg,
                    controlnet_strength=open_cn_strength,
                    sampler=open_sampler,   scheduler=open_scheduler),
        }

        lora_cfg = {
            0: {"path": lora_closed_path or "",
                "strength_model": lora_closed_strength_model,
                "strength_clip":  lora_closed_strength_clip},
            1: {"path": lora_half_path or "",
                "strength_model": lora_half_strength_model,
                "strength_clip":  lora_half_strength_clip},
            2: {"path": lora_open_path or "",
                "strength_model": lora_open_strength_model,
                "strength_clip":  lora_open_strength_clip},
        }

        pos_conds_ext: Dict[int, Any] = {}
        if cond_closed is not None: pos_conds_ext[0] = cond_closed
        if cond_half   is not None: pos_conds_ext[1] = cond_half
        if cond_open   is not None: pos_conds_ext[2] = cond_open

        output_dir           = get_comfy_output_dir()
        stem                 = Path(video_output_filename).stem or "lipsync_output"
        final_path, filename = unique_filename(output_dir, stem, ".mp4")
        debug_dir = ""

        print(f"[LipSync V5.5] Output → {final_path}")
        for mt, lcfg in lora_cfg.items():
            p = lcfg["path"]
            if p:
                import os as _os
                print(f"  LoRA [{MOUTH_TYPE_NAMES[mt]}]: {_os.path.basename(p)} "
                      f"model={lcfg['strength_model']} clip={lcfg['strength_clip']}")

        with tempfile.TemporaryDirectory(prefix="lipsync_zturbo_") as work_dir:
            store = DiskStore(Path(work_dir))

            # ── Allosaurus ────────────────────────────────────────────────────
            wav_path       = comfy_audio_to_wav(audio, work_dir)
            audio_orig_wav = save_audio_original(audio, work_dir)
            detector       = AllosaurusDetector(wav_path, fps, lang_id)
            phonemes, mouth_types_list = detector.detect()
            n_frames = len(phonemes)
            print(f"[F0] {n_frames} frames @ {fps}fps | {detector._duration:.2f}s")
            print(f"[F0] closed={mouth_types_list.count(0)} "
                  f"half={mouth_types_list.count(1)} "
                  f"open={mouth_types_list.count(2)}")

            # ── Salva frames de entrada ───────────────────────────────────────
            src_total = images.shape[0]
            for i in range(n_frames):
                src_idx  = min(int(i / fps * source_fps), src_total - 1)
                frame_np = t2np(images[src_idx])
                store.save_orig(i, frame_np)
                del frame_np
            del images
            cuda_cleanup()

            # ── Fase 0 ────────────────────────────────────────────────────────
            # Florence-2: detecta personagem em todos os frames (single-thread,
            # antes dos workers YOLO para evitar contenção de GPU)
            character_bboxes = None
            if enable_character_detect and character_query and character_query.strip():
                print(f"[LipSync] Florence-2 ativado — detectando personagem: '{character_query}'")
                character_bboxes = build_character_bboxes(
                    store, n_frames,
                    query=character_query.strip(),
                    margin=character_margin,
                    florence_model_id=florence_model_id,
                )

            n_workers = compute_workers(vram_safety_margin_mb)
            run_phase0(
                store, n_frames,
                detection_model_path,
                vram_safety_margin_mb,
                mask_dilation, mask_blur,
                remove_mouth, upscale_crop_face,
                mouth_conf, n_workers,
                character_bboxes=character_bboxes,
            )

            # ── Fase 1 (+ overlap) ────────────────────────────────────────────
            if enable_overlap:
                ready_queue     = Queue()
                canonical_cache: Dict = {}

                overlap_threads, stop_event = launch_overlap_workers(
                    ready_queue, store, mouth_types_list, detection_model_path,
                    compose_feather_px, canonical_cache,
                    vram_safety_margin_mb, upscale_crop_face, mouth_conf,
                    mouth_padding_per_type=mouth_padding_per_type,
                    mouth_brightness_per_type=mouth_brightness_per_type,
                    use_rembg=use_rembg,
                    use_open_for_half=use_open_for_half,
                    half_open_height_scale=half_open_height_scale,
                )

                with torch.inference_mode():
                    run_phase1(
                        store, phonemes, mouth_types_list, n_frames,
                        cfg_per_type=cfg_per_type,
                        model_obj=model, model_patch_obj=model_patch,
                        clip_obj=clip, vae_obj=vae,
                        pos_conds_ext=pos_conds_ext,
                        sim_threshold=sim_threshold,
                        vram_safety_mb=vram_safety_margin_mb,
                        ready_queue=ready_queue,
                        upscale_crop_face=upscale_crop_face,
                        lora_cfg=lora_cfg,
                        hed_detector_mode=hed_detector_mode,
                        motion_variance_factor=motion_variance_factor,
                        detection_model_path=detection_model_path,
                        conf_mouth=mouth_conf,
                        verify_generated_mouth=verify_generated_mouth,
                        mouth_regen_attempts=mouth_regen_attempts,
                        use_open_for_half=use_open_for_half,
                    )

                timeout_s = n_frames * 5 + 120
                for t in overlap_threads:
                    t.join(timeout=timeout_s)
                    if t.is_alive():
                        print(f"[Overlap] Worker {t.name} não finalizou.")
                stop_event.set()
                #smooth_mouth_tracks(store, mouth_types_list, n_frames)
                n_workers_f3 = compute_workers(vram_safety_margin_mb)
                run_phase3(store, n_frames, compose_feather_px, n_workers_f3)
                ensure_all_outputs(store, n_frames, compose_feather_px)

            else:
                with torch.inference_mode():
                    run_phase1(
                        store, phonemes, mouth_types_list, n_frames,
                        cfg_per_type=cfg_per_type,
                        model_obj=model, model_patch_obj=model_patch,
                        clip_obj=clip, vae_obj=vae,
                        pos_conds_ext=pos_conds_ext,
                        sim_threshold=sim_threshold,
                        vram_safety_mb=vram_safety_margin_mb,
                        ready_queue=None,
                        upscale_crop_face=upscale_crop_face,
                        lora_cfg=lora_cfg,
                        hed_detector_mode=hed_detector_mode,
                        motion_variance_factor=motion_variance_factor,
                        detection_model_path=detection_model_path,
                        conf_mouth=mouth_conf,
                        verify_generated_mouth=verify_generated_mouth,
                        mouth_regen_attempts=mouth_regen_attempts,
                        use_open_for_half=use_open_for_half,
                    )

                n_workers_f2 = compute_workers(vram_safety_margin_mb)
                run_phase2(
                    store, mouth_types_list, n_frames, detection_model_path,
                    n_workers_f2, upscale_crop_face, mouth_conf,
                    mouth_padding_per_type=mouth_padding_per_type,
                    mouth_brightness_per_type=mouth_brightness_per_type,
                    use_rembg=use_rembg,
                    use_open_for_half=use_open_for_half,
                    half_open_height_scale=half_open_height_scale,
                )

                n_workers_f3 = compute_workers(vram_safety_margin_mb)
                run_phase3(store, n_frames, compose_feather_px, n_workers_f3)

            # ── Vídeo final ───────────────────────────────────────────────────
            video_ok = False
            try:
                build_video_with_audio(
                    store.output_dir, audio_orig_wav, n_frames, fps, final_path)
                print(f"[Video] Salvo: {final_path}")
                video_ok = True
            except Exception as exc:
                print(f"[Video] ERRO: {exc}")
                filename = ""; final_path = ""

            if save_debug_folder:
                debug_base = Path(output_dir) / f"{stem}_debug"
                debug_path = debug_base
                counter = 1
                while debug_path.exists():
                    debug_path = Path(output_dir) / f"{stem}_debug_{counter:04d}"
                    counter += 1
                shutil.copytree(store.work_dir, debug_path)
                debug_dir = str(debug_path)
                print(f"[Debug] Pasta salva: {debug_dir}")

        elapsed = time.perf_counter() - t_start
        print(f"\n[Done] {elapsed:.1f}s | {elapsed / n_frames:.2f}s/frame")

        video_info = {
            "video_path": final_path,
            "fps": fps,
            "n_frames": n_frames,
            "debug_dir": debug_dir,
        }

        if video_ok and filename:
            return {
                "ui": {
                    "videos": [{"filename": filename, "subfolder": "", "type": "output"}],
                    "animated": [False],
                },
                "result": (final_path, video_info, debug_dir),
            }
        return {"ui": {"videos": []}, "result": ("", video_info, debug_dir)}


# ===========================================================================
# Registro ComfyUI
# ===========================================================================

NODE_CLASS_MAPPINGS = {
    "LipSyncPipelineV6": LipSyncZTurboPipeline,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LipSyncPipelineV6": "LipSync Pipeline V6",
}
