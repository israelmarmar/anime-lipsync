"""
phase0.py — Fase 0: preprocessamento paralelo
==============================================

Fluxo por frame (sem personagem):
  frame → YOLO face → face_bbox → YOLO boca (no crop upscalado) → inpaint

Fluxo por frame (com personagem via Florence-2):
  frame → character_bbox (pré-calculado) → crop do personagem
        → YOLO face no crop → face_bbox (coordenadas absolutas no frame)
        → YOLO boca (no crop upscalado da face) → inpaint
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .constants import _OOM_MAX_RETRIES, _OOM_BACKOFF_BASE
from .store import DiskStore, VramGuard
from .utils import build_mask, cv2_inpaint, cuda_cleanup, vram_info


_TARGET_SIDE_PX = 512  # lado alvo para upscale dinamico


def _dynamic_upscale(fH: int, fW: int, upscale_crop_face: float) -> Tuple[int, int]:
    """
    Calcula (up_W, up_H) de forma que o maior lado do crop seja exatamente
    _TARGET_SIDE_PX, garantindo precisao maxima para o YOLO de boca.
    O fator upscale_crop_face e usado somente como fallback minimo caso
    o crop ja seja maior que o alvo.
    """
    max_side = max(fH, fW)
    if max_side >= _TARGET_SIDE_PX:
        scale = upscale_crop_face
    else:
        scale = _TARGET_SIDE_PX / max_side
        scale = max(scale, upscale_crop_face)
    up_W = max(fW, int(round(fW * scale)))
    up_H = max(fH, int(round(fH * scale)))
    return up_W, up_H



def _run_yolo(model, img, conf):
    with torch.no_grad():
        results = model(img, verbose=False, conf=conf)
    boxes = results[0].boxes
    xyxy = confs = None
    if boxes is not None and len(boxes) > 0:
        xyxy  = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
    del results, boxes
    return xyxy, confs


def preprocess_frame(
    idx: int,
    store: DiskStore,
    face_model_path: str,
    mouth_model_path: str,
    guard: VramGuard,
    mask_dilation: int,
    mask_blur: int,
    remove_mouth: bool = True,
    upscale_crop_face: float = 2.0,
    conf_face: float = 0.25,
    conf_mouth: float = 0.1,
    face_margin: float = 0.10,
    character_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> int:
    """
    Preprocessa um frame.

    character_bbox : (x1,y1,x2,y2) do personagem no frame completo,
                     pré-calculado pela Florence-2. Quando fornecido,
                     o YOLO de face só busca dentro deste crop.
    """
    from ultralytics import YOLO

    last_exc = None
    for attempt in range(_OOM_MAX_RETRIES + 1):
        if attempt > 0:
            wait = _OOM_BACKOFF_BASE ** attempt
            print(f"[Preproc {idx}] Retry {attempt} aguardando {wait:.1f}s")
            cuda_cleanup()
            time.sleep(wait)
        guard.acquire()
        try:
            img_np = store.load_orig(idx)
            H, W   = img_np.shape[:2]

            face_bbox_m = None
            mask_np     = np.zeros((H, W), dtype=np.uint8)
            inpainted   = img_np.copy()

            # ── Região de busca de face ───────────────────────────────────────
            # Com personagem: restringe ao crop da Florence-2.
            # Sem personagem: frame inteiro.
            if character_bbox is not None:
                cx1, cy1, cx2, cy2 = character_bbox
                search_img    = img_np[cy1:cy2, cx1:cx2]
                search_offset = (cx1, cy1)          # offset para remap de coords
            else:
                search_img    = img_np
                search_offset = (0, 0)

            sH, sW = search_img.shape[:2]

            if sH > 0 and sW > 0:
                face_model      = YOLO(face_model_path)
                xyxy_f, confs_f = _run_yolo(face_model, search_img, conf_face)
                del face_model

                if xyxy_f is not None:
                    best               = int(np.argmax(confs_f))
                    fx1, fy1, fx2, fy2 = xyxy_f[best].astype(int)

                    # Remap para coordenadas absolutas no frame
                    ox, oy = search_offset
                    fx1 += ox; fy1 += oy; fx2 += ox; fy2 += oy

                    mx  = int((fx2 - fx1) * face_margin)
                    my  = int((fy2 - fy1) * face_margin)
                    bx1 = max(0, fx1 - mx); by1 = max(0, fy1 - my)
                    bx2 = min(W, fx2 + mx); by2 = min(H, fy2 + my)
                    face_bbox_m = (bx1, by1, bx2, by2)

                    if remove_mouth:
                        face_crop = img_np[by1:by2, bx1:bx2].copy()
                        fH, fW = face_crop.shape[:2]
                        if fH > 0 and fW > 0:
                            up_W, up_H = _dynamic_upscale(fH, fW, upscale_crop_face)
                            face_up = cv2.resize(face_crop, (up_W, up_H),
                                                 interpolation=cv2.INTER_LANCZOS4)

                            mouth_model     = YOLO(mouth_model_path)
                            xyxy_m, confs_m = _run_yolo(mouth_model, face_up, conf_mouth)
                            del mouth_model

                            if xyxy_m is not None:
                                mb = int(np.argmax(confs_m))
                                umx1, umy1, umx2, umy2 = xyxy_m[mb].astype(int)
                                umx1 = max(0, umx1); umy1 = max(0, umy1)
                                umx2 = min(up_W, umx2); umy2 = min(up_H, umy2)

                                sx  = fW / up_W; sy = fH / up_H
                                mx1 = max(0,  int(umx1 * sx))
                                my1 = max(0,  int(umy1 * sy))
                                mx2 = min(fW, int(umx2 * sx))
                                my2 = min(fH, int(umy2 * sy))

                                mask_crop    = build_mask(fH, fW, (mx1, my1, mx2, my2),
                                                          dilation=mask_dilation, blur=0)
                                face_bgr     = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                                face_inp_bgr = cv2_inpaint(face_bgr, mask_crop)
                                face_inp     = cv2.cvtColor(face_inp_bgr, cv2.COLOR_BGR2RGB)
                                del face_bgr, face_inp_bgr

                                mask_full = build_mask(fH, fW, (mx1, my1, mx2, my2),
                                                       dilation=mask_dilation, blur=mask_blur)
                                alpha  = mask_full.astype(np.float32)[:, :, None] / 255.0
                                region = img_np[by1:by2, bx1:bx2].astype(np.float32)
                                inpainted[by1:by2, bx1:bx2] = (
                                    face_inp.astype(np.float32) * alpha
                                    + region * (1.0 - alpha)
                                ).clip(0, 255).astype(np.uint8)
                                del face_inp, alpha, region
                                mask_np[by1:by2, bx1:bx2] = mask_full
                                del mask_crop, mask_full

                            del face_up
                        del face_crop

                del xyxy_f, confs_f

            fmask_np = np.zeros((H, W), dtype=np.uint8)
            if face_bbox_m is not None:
                bx1, by1, bx2, by2 = face_bbox_m
                fmask_np[by1:by2, bx1:bx2] = 255

            store.save_nomouth(idx, inpainted)
            store.save_mmask(idx, mask_np)
            store.save_fmask(idx, fmask_np)
            if face_bbox_m is not None:
                store.save_face_bbox(idx, *[int(v) for v in face_bbox_m])
            if character_bbox is not None:
                store.save_character_bbox(idx, *[int(v) for v in character_bbox])

            del img_np, mask_np, fmask_np, inpainted
            return idx

        except RuntimeError as exc:
            if any(k in str(exc).lower() for k in
                   ("allocation on device", "out of memory", "cuda error")):
                last_exc = exc
                guard.reduce()
                cuda_cleanup()
            else:
                raise
        finally:
            guard.release()
            cuda_cleanup()

    raise RuntimeError(f"Frame {idx}: retries esgotados. {last_exc}")


def run_phase0(
    store: DiskStore,
    n_frames: int,
    face_model_path: str,
    mouth_model_path: str,
    vram_safety_mb: int,
    mask_dilation: int,
    mask_blur: int,
    remove_mouth: bool,
    upscale_crop_face: float,
    mouth_conf: float,
    n_workers: int,
    character_bboxes: Optional[Dict[int, Optional[Tuple[int, int, int, int]]]] = None,
) -> None:
    """
    character_bboxes : dict {frame_idx: (x1,y1,x2,y2) | None}
                       produzido por character_detect.build_character_bboxes().
                       None ou dict vazio = sem seleção de personagem.
    """
    from .constants import _VRAM_PER_WORKER_MB

    guard    = VramGuard(n_workers, vram_safety_mb, _VRAM_PER_WORKER_MB)
    log_step = max(1, n_frames // 10)
    completed = 0
    char_mode = bool(character_bboxes)

    print(f"[F0] {'YOLO face+boca+inpaint' if remove_mouth else 'YOLO somente face'} "
          f"— {n_frames} frames, {n_workers} workers, "
          f"upscale=auto(min={upscale_crop_face:.1f}x, alvo={_TARGET_SIDE_PX}px) conf={mouth_conf}"
          + (" [modo personagem Florence-2]" if char_mode else "") + "...")

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(
                preprocess_frame, i, store,
                face_model_path, mouth_model_path,
                guard, mask_dilation, mask_blur,
                remove_mouth, upscale_crop_face,
                0.25, mouth_conf,
                0.10,
                character_bboxes.get(i) if char_mode else None,
            ): i
            for i in range(n_frames)
        }
        for future in as_completed(futures):
            orig_idx = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[F0] FALLBACK frame {orig_idx}: {exc}")
                orig = store.load_orig(orig_idx)
                store.save_nomouth(orig_idx, orig)
                store.save_mmask(orig_idx, np.zeros(orig.shape[:2], dtype=np.uint8))
                store.save_fmask(orig_idx, np.zeros(orig.shape[:2], dtype=np.uint8))
            completed += 1
            if completed % log_step == 0 or completed == n_frames:
                print(f"[F0] {completed}/{n_frames} | {vram_info()}")

    cuda_cleanup()
    print(f"[F0] Concluído. {vram_info()}")
