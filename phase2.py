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
from .phase1 import _detect_anatomical_mouth

try:
    from rembg import remove as _rembg_remove
    _REMBG_AVAILABLE = True
except ImportError:
    _REMBG_AVAILABLE = False

_BEN2_LOCK = threading.Lock()
_BEN2_NODE = None
_BEN2_WARNED = False


def _load_easy_image_rembg_node():
    global _BEN2_NODE
    with _BEN2_LOCK:
        if _BEN2_NODE is not None:
            return _BEN2_NODE

        try:
            from nodes import NODE_CLASS_MAPPINGS
        except Exception as exc:
            raise ImportError("NODE_CLASS_MAPPINGS do ComfyUI indisponível.") from exc

        cls = NODE_CLASS_MAPPINGS.get("easy imageRemBg")
        if cls is None:
            raise KeyError("Custom node 'easy imageRemBg' não está carregado.")

        _BEN2_NODE = cls()
        print("[F2] BEN2 via easy imageRemBg carregado.")
        return _BEN2_NODE


def _tensor_to_np_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.nan_to_num(arr).clip(0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def _easy_ben2_rgba(rgb_crop: np.ndarray,
                    refine_foreground: bool = False) -> Optional[np.ndarray]:
    global _BEN2_WARNED
    try:
        node = _load_easy_image_rembg_node()
        images = torch.from_numpy(rgb_crop.astype(np.float32) / 255.0).unsqueeze(0)
        with _BEN2_LOCK:
            out = node.remove(
                rem_mode="BEN2",
                image_output="Hide",
                save_prefix="ComfyUI",
                torchscript_jit=False,
                add_background="none",
                refine_foreground=refine_foreground,
                images=images,
            )

        out_image = out[0] if isinstance(out, (tuple, list)) else out["result"][0]
        out_np = _tensor_to_np_image(out_image)
        if out_np.ndim == 2:
            out_np = np.dstack([rgb_crop, out_np])
        elif out_np.shape[-1] == 3:
            mask_np = None
            if isinstance(out, (tuple, list)) and len(out) > 1:
                mask_np = _tensor_to_np_image(out[1])
            if mask_np is None:
                alpha = np.full(rgb_crop.shape[:2], 255, dtype=np.uint8)
            else:
                alpha = mask_np[..., 0] if mask_np.ndim == 3 else mask_np
            out_np = np.dstack([out_np[:, :, :3], alpha])
        elif out_np.shape[-1] > 4:
            out_np = out_np[:, :, :4]

        if out_np.shape[:2] != rgb_crop.shape[:2]:
            out_np = cv2.resize(out_np, (rgb_crop.shape[1], rgb_crop.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
        return out_np.astype(np.uint8)
    except Exception as exc:
        if not _BEN2_WARNED:
            print(f"[F2] easy imageRemBg BEN2 indisponível; usando fallback local/rembg: {exc}")
            _BEN2_WARNED = True
        return None


def _easy_ben2_alpha(rgb_crop: np.ndarray,
                     refine_foreground: bool = False) -> Optional[np.ndarray]:
    rgba = _easy_ben2_rgba(rgb_crop, refine_foreground=refine_foreground)
    if rgba is None:
        return None
    return rgba[:, :, 3].astype(np.float32) / 255.0


def _expand_bbox(bx1: int, by1: int, bx2: int, by2: int,
                 fW: int, fH: int, mouth_type: int,
                 user_padding: int) -> Tuple[int, int, int, int]:
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    if mouth_type == 0:
        auto_x = max(1, int(round(bw * 0.03)))
        auto_y = max(1, int(round(bh * 0.04)))
    elif mouth_type == 1:
        auto_x = max(1, int(round(bw * 0.04)))
        auto_y = max(1, int(round(bh * 0.08)))
    else:
        auto_x = max(1, int(round(bw * 0.05)))
        auto_y = max(1, int(round(bh * 0.10)))

    pad_x = auto_x + user_padding
    pad_y = auto_y + user_padding
    return (
        max(0, bx1 - pad_x),
        max(0, by1 - pad_y),
        min(fW, bx2 + pad_x),
        min(fH, by2 + pad_y),
    )


def _ellipse_mask(h: int, w: int, scale_x: float, scale_y: float,
                  blur_divisor: int = 12) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (max(1, int(w * scale_x)), max(1, int(h * scale_y)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    k = max(3, (min(h, w) // blur_divisor) | 1)
    return cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0


def _open_mouth_rgba(mouth_crop: np.ndarray, mouth_type: int,
                     use_rembg: bool = False) -> np.ndarray:
    ben2_rgba = _easy_ben2_rgba(mouth_crop) if use_rembg else None
    if ben2_rgba is not None:
        return ben2_rgba

    if use_rembg and _REMBG_AVAILABLE:
        rembg_rgba = np.array(
            _rembg_remove(Image.fromarray(mouth_crop)).convert("RGBA"), dtype=np.uint8)
        rembg_alpha = rembg_rgba[:, :, 3].astype(np.float32) / 255.0
    else:
        rembg_alpha = np.ones(mouth_crop.shape[:2], dtype=np.float32)

    h, w = mouth_crop.shape[:2]
    hsv = cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    r = mouth_crop[:, :, 0].astype(np.int16)
    g = mouth_crop[:, :, 1].astype(np.int16)
    b = mouth_crop[:, :, 2].astype(np.int16)
    red_bias = ((r > g + 8) & (r > b + 4)).astype(np.float32)

    dark_or_color = np.maximum.reduce([
        np.clip((0.60 - gray) / 0.34, 0.0, 1.0),
        np.clip((sat - 0.30) / 0.38, 0.0, 1.0),
        red_bias * np.clip((r.astype(np.float32) - 80.0) / 140.0, 0.0, 1.0),
    ])

    edges = cv2.Canny((gray * 255).astype(np.uint8), 35, 110)
    edge_r = max(1, min(h, w) // 18)
    edge_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_r * 2 + 1, edge_r * 2 + 1))
    edge_mask = cv2.dilate(edges, edge_k, iterations=1).astype(np.float32) / 255.0

    ellipse = _ellipse_mask(h, w, 0.43, 0.39 if mouth_type == 1 else 0.43)
    edge_support = cv2.GaussianBlur(dark_or_color, (5, 5), 0)
    alpha = np.maximum(rembg_alpha * dark_or_color, edge_mask * edge_support * 0.85)
    alpha = np.maximum(alpha, dark_or_color * 0.65)
    alpha *= ellipse

    alpha[alpha < 0.22] = 0.0
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    alpha[alpha < 0.18] = 0.0
    alpha = np.clip(alpha, 0.0, 1.0)

    rgb = mouth_crop.copy()
    return np.dstack([rgb, (alpha * 255).astype(np.uint8)])


def _face_motion_signature(img_np: np.ndarray, face_bbox) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if face_bbox is None:
        return None
    x1, y1, x2, y2 = face_bbox
    crop = img_np[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 120)

    mask = np.ones((128, 128), dtype=np.uint8) * 255
    # Ignora a regiao central inferior, onde a boca original pode variar.
    mask[70:118, 32:96] = 0
    gray = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.bitwise_and(edges, edges, mask=mask)
    return gray, edges


def _face_signature_stable(prev_sig, sig,
                           gray_threshold: float = 0.035,
                           edge_threshold: float = 0.085) -> bool:
    if prev_sig is None or sig is None:
        return False
    prev_gray, prev_edges = prev_sig
    gray, edges = sig
    gray_diff = float(np.mean(cv2.absdiff(prev_gray, gray)) / 255.0)
    edge_diff = float(np.mean(cv2.absdiff(prev_edges, edges)) / 255.0)
    return gray_diff <= gray_threshold and edge_diff <= edge_threshold


def _mouth_cached_pos_to_frame(pos: dict,
                               frame_shape: Tuple[int, int],
                               face_bbox=None) -> Tuple[int, int, int, int]:
    H_fr, W_fr = frame_shape
    if face_bbox is not None and all(k in pos for k in ("rx", "ry", "rw", "rh")):
        fx1, fy1, fx2, fy2 = face_bbox
        face_w = max(1, int(fx2) - int(fx1))
        face_h = max(1, int(fy2) - int(fy1))
        ax = int(fx1) + int(round(float(pos.get("rx", 0.0)) * face_w))
        ay = int(fy1) + int(round(float(pos.get("ry", 0.0)) * face_h))
        aw = int(round(float(pos.get("rw", 0.0)) * face_w))
        ah = int(round(float(pos.get("rh", 0.0)) * face_h))
    elif face_bbox is not None and all(k in pos for k in ("ref_fx1", "ref_fy1", "ref_fw", "ref_fh")):
        fx1, fy1, fx2, fy2 = face_bbox
        face_w = max(1, int(fx2) - int(fx1))
        face_h = max(1, int(fy2) - int(fy1))
        ref_fx1 = int(pos.get("ref_fx1", 0))
        ref_fy1 = int(pos.get("ref_fy1", 0))
        ref_fw = max(1, int(pos.get("ref_fw", 1)))
        ref_fh = max(1, int(pos.get("ref_fh", 1)))
        still_tol = max(4, int(round(max(ref_fw, ref_fh) * 0.03)))
        face_still = (
            abs(int(fx1) - ref_fx1) <= still_tol and
            abs(int(fy1) - ref_fy1) <= still_tol and
            abs(face_w - ref_fw) <= still_tol and
            abs(face_h - ref_fh) <= still_tol
        )
        if face_still:
            ax = int(pos.get("ax", 0))
            ay = int(pos.get("ay", 0))
            aw = int(pos.get("aw", 0))
            ah = int(pos.get("ah", 0))
        else:
            sx = face_w / ref_fw
            sy = face_h / ref_fh
            ax = int(fx1) + int(round(int(pos.get("gx", 0)) * sx))
            ay = int(fy1) + int(round(int(pos.get("gy", 0)) * sy))
            aw = int(round(int(pos.get("gw", 0)) * sx))
            ah = int(round(int(pos.get("gh", 0)) * sy))
    elif face_bbox is not None and all(k in pos for k in ("gx", "gy", "gw", "gh")):
        fx1, fy1, fx2, fy2 = face_bbox
        face_w = max(1, int(fx2) - int(fx1))
        face_h = max(1, int(fy2) - int(fy1))
        src_w = int(pos.get("generated_face_w", 0)) or face_w
        src_h = int(pos.get("generated_face_h", 0)) or face_h
        sx = face_w / max(1, src_w)
        sy = face_h / max(1, src_h)
        ax = int(fx1) + int(round(int(pos.get("gx", 0)) * sx))
        ay = int(fy1) + int(round(int(pos.get("gy", 0)) * sy))
        aw = int(round(int(pos.get("gw", 0)) * sx))
        ah = int(round(int(pos.get("gh", 0)) * sy))
    else:
        ax = int(pos.get("ax", 0))
        ay = int(pos.get("ay", 0))
        aw = int(pos.get("aw", 0))
        ah = int(pos.get("ah", 0))

    ax = max(0, min(W_fr - 1, ax))
    ay = max(0, min(H_fr - 1, ay))
    aw = min(max(1, aw), W_fr - ax)
    ah = min(max(1, ah), H_fr - ay)
    return ax, ay, aw, ah


def _detect_mouth(face_np: np.ndarray,
                  face_meta: dict,
                  face_bbox,
                  nomouth_shape: Tuple[int, int],
                  detection_model,
                  upscale_crop_face: float,
                  conf_mouth: float,
                  mouth_type: int = 0,
                  mouth_padding_per_type: Dict[int, int] = None,
                  mouth_brightness_per_type: Dict[int, float] = None,
                  use_rembg: bool = False) -> Optional[Tuple[np.ndarray, int, int, int, int]]:
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

    # Seleciona somente bocas na região anatômica entre nariz e queixo.
    mouth_det, up_W, up_H, anatomy_msg = _detect_anatomical_mouth(
        face_np, detection_model, conf_mouth, upscale_crop_face)

    if mouth_det is None:
        print(f"  [F2] Boca rejeitada por posição anatômica: {anatomy_msg}")
        return None

    (bx1u, by1u, bx2u, by2u), _ = mouth_det
    bx1u = max(0, bx1u); by1u = max(0, by1u)
    bx2u = min(up_W, bx2u); by2u = min(up_H, by2u)

    if bx2u <= bx1u or by2u <= by1u:
        return None

    # Remap upscale → face_np (1:1 com frame)
    sx = fW / up_W; sy = fH / up_H
    bx1 = max(0,  int(bx1u * sx)); by1 = max(0,  int(by1u * sy))
    bx2 = min(fW, int(bx2u * sx)); by2 = min(fH, int(by2u * sy))

    if bx2 <= bx1 or by2 <= by1:
        return None

    bx1, by1, bx2, by2 = _expand_bbox(
        bx1, by1, bx2, by2, fW, fH, mouth_type, mouth_padding)

    mouth_crop = face_np[by1:by2, bx1:bx2].copy()

    # Ajuste de brilho do crop da boca
    if abs(mouth_brightness - 1.0) > 0.001:
        mouth_crop = np.clip(
            mouth_crop.astype(np.float32) * mouth_brightness, 0, 255
        ).astype(np.uint8)

    if mouth_type == 0:
        # ── neutral_closed: preserva linhas, não uma mancha de pele ──────────
        # Para boca fechada, preencher a elipse inteira carrega pele gerada com
        # tom diferente. Aqui o alpha fica restrito a linhas/escuridão dos lábios.
        ben2_rgba = _easy_ben2_rgba(mouth_crop) if use_rembg else None
        if ben2_rgba is not None:
            mouth_rgba = ben2_rgba
            del mouth_crop

            target_w = max(1, int(fx2) - int(fx1))
            target_h = max(1, int(fy2) - int(fy1))
            frame_sx = target_w / max(1, fW)
            frame_sy = target_h / max(1, fH)

            abs_x = max(0, min(W_fr - 1, int(fx1) + int(round(bx1 * frame_sx))))
            abs_y = max(0, min(H_fr - 1, int(fy1) + int(round(by1 * frame_sy))))
            abs_w = min(max(1, int(round((bx2 - bx1) * frame_sx))), W_fr - abs_x)
            abs_h = min(max(1, int(round((by2 - by1) * frame_sy))), H_fr - abs_y)

            return mouth_rgba, abs_x, abs_y, abs_w, abs_h

        ch, cw = mouth_crop.shape[:2]

        gray = cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1].astype(np.float32) / 255.0
        gray_eq = cv2.equalizeHist(gray)

        lo = max(20, int(cw * 0.5))
        hi = max(60, int(cw * 1.5))
        edges = cv2.Canny(gray_eq, lo, hi)
        dil_r = max(1, min(ch, cw) // 12)
        dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_r * 2 + 1, dil_r * 2 + 1))
        edge_mask = cv2.dilate(edges, dil_k, iterations=1).astype(np.float32) / 255.0
        edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0)

        gray_f = gray.astype(np.float32) / 255.0
        dark_line = np.clip((0.58 - gray_f) / 0.28, 0.0, 1.0)
        color_line = np.clip((sat - 0.24) / 0.36, 0.0, 1.0)
        line_mask = np.maximum(edge_mask * 0.95, np.maximum(dark_line, color_line) * 0.70)

        support = np.zeros((ch, cw), dtype=np.uint8)
        cx_e, cy_e = cw // 2, ch // 2
        axes = (max(1, int(cw * 0.48)), max(1, int(ch * 0.30)))
        cv2.ellipse(support, (cx_e, cy_e), axes, 0, 0, 360, 255, -1)
        support_k = max(3, (min(ch, cw) // 14) | 1)
        support_f = cv2.GaussianBlur(support, (support_k, support_k), 0).astype(np.float32) / 255.0

        ben2_alpha = _easy_ben2_alpha(mouth_crop) if use_rembg else None
        if ben2_alpha is not None:
            support_f = np.maximum(support_f, ben2_alpha * 0.85)

        mask_f = line_mask * support_f
        mask_f[mask_f < 0.18] = 0.0
        mask_f = cv2.GaussianBlur(mask_f, (3, 3), 0)
        mask_f[mask_f < 0.12] = 0.0
        mask_f = np.clip(mask_f, 0.0, 1.0)

        alpha8 = (mask_f * 255).clip(0, 255).astype(np.uint8)
        mouth_rgba = np.dstack([mouth_crop, alpha8])
    else:
        # ── half_open / fully_open: combina YOLO + rembg + máscara de cor ───
        # O BEN2/rembg sozinho às vezes preserva pele ao redor dos lábios; a máscara
        # abaixo restringe o alpha a regiões escuras/coloridas típicas da boca.
        mouth_rgba = _open_mouth_rgba(mouth_crop, mouth_type, use_rembg=use_rembg)

    del mouth_crop

    target_w = max(1, int(fx2) - int(fx1))
    target_h = max(1, int(fy2) - int(fy1))
    frame_sx = target_w / max(1, fW)
    frame_sy = target_h / max(1, fH)

    abs_x = max(0, min(W_fr - 1, int(fx1) + int(round(bx1 * frame_sx))))
    abs_y = max(0, min(H_fr - 1, int(fy1) + int(round(by1 * frame_sy))))
    abs_w = min(max(1, int(round((bx2 - bx1) * frame_sx))), W_fr - abs_x)
    abs_h = min(max(1, int(round((by2 - by1) * frame_sy))), H_fr - abs_y)

    return mouth_rgba, abs_x, abs_y, abs_w, abs_h


def smooth_mouth_tracks(store: DiskStore,
                        mouth_types: List[int],
                        n_frames: int,
                        center_alpha: float = 0.45,
                        size_alpha: float = 0.55,
                        max_rel_jump: float = 0.20,
                        group_center_strength: float = 0.05,
                        stable_face_px: float = 0.75,
                        stable_mouth_alpha: float = 0.0,
                        mouth_deadband_px: float = 2.0) -> None:
    """
    Suaviza a posição da boca em coordenadas relativas ao bbox da face.

    Isso reduz jitter de detecção sem prender a boca em coordenadas absolutas:
    quando a cabeça se move, o bbox da face muda e a boca acompanha.
    """
    import statistics

    centers_by_group: Dict[int, List[Tuple[float, float]]] = {}
    for q in range(n_frames):
        if not store.has_mouth_frame(q):
            continue
        face_bbox = store.load_face_bbox(q)
        if face_bbox is None:
            continue
        rgba, ax, ay, aw, ah = store.load_mouth_frame(q)
        if (rgba.shape[0] <= 4 and rgba.shape[1] <= 4) or aw <= 0 or ah <= 0:
            continue
        meta = store.load_face_frame_meta(q) if store.has_face_frame(q) else {}
        fg = int(meta.get("face_group", 0))
        fx1, fy1, fx2, fy2 = face_bbox
        fw = max(1, fx2 - fx1)
        fh = max(1, fy2 - fy1)
        centers_by_group.setdefault(fg, []).append((
            (ax + aw * 0.5 - fx1) / fw,
            (ay + ah * 0.5 - fy1) / fh,
        ))

    group_centers = {
        fg: (
            statistics.median(c[0] for c in centers),
            statistics.median(c[1] for c in centers),
        )
        for fg, centers in centers_by_group.items()
        if centers
    }

    prev = None
    adjusted = 0

    for q in range(n_frames):
        if not store.has_mouth_frame(q):
            prev = None
            continue

        face_bbox = store.load_face_bbox(q)
        if face_bbox is None:
            prev = None
            continue
        face_sig = _face_motion_signature(store.load_orig(q), face_bbox)

        rgba, ax, ay, aw, ah = store.load_mouth_frame(q)
        if (rgba.shape[0] <= 4 and rgba.shape[1] <= 4) or aw <= 0 or ah <= 0:
            prev = None
            continue

        meta = store.load_face_frame_meta(q) if store.has_face_frame(q) else {}
        fg = int(meta.get("face_group", 0))
        mt = mouth_types[q]

        fx1, fy1, fx2, fy2 = face_bbox
        fw = max(1, fx2 - fx1)
        fh = max(1, fy2 - fy1)
        face_cx = (fx1 + fx2) * 0.5
        face_cy = (fy1 + fy2) * 0.5
        cx_rel = (ax + aw * 0.5 - fx1) / fw
        cy_rel = (ay + ah * 0.5 - fy1) / fh
        w_rel = aw / fw
        h_rel = ah / fh

        if fg in group_centers:
            gcx, gcy = group_centers[fg]
            cx_rel = gcx * group_center_strength + cx_rel * (1.0 - group_center_strength)
            cy_rel = gcy * group_center_strength + cy_rel * (1.0 - group_center_strength)

        if prev is not None and prev["fg"] == fg:
            dx = abs(cx_rel - prev["cx"])
            dy = abs(cy_rel - prev["cy"])
            if dx <= max_rel_jump and dy <= max_rel_jump:
                cx_rel = prev["cx"] * (1.0 - center_alpha) + cx_rel * center_alpha
                cy_rel = prev["cy"] * (1.0 - center_alpha) + cy_rel * center_alpha
                if prev["mt"] == mt:
                    w_rel = prev["w"] * (1.0 - size_alpha) + w_rel * size_alpha
                    h_rel = prev["h"] * (1.0 - size_alpha) + h_rel * size_alpha

        new_aw = max(1, int(round(w_rel * fw)))
        new_ah = max(1, int(round(h_rel * fh)))
        new_ax = int(round(fx1 + cx_rel * fw - new_aw * 0.5))
        new_ay = int(round(fy1 + cy_rel * fh - new_ah * 0.5))

        if prev is not None and prev["fg"] == fg:
            face_delta = max(
                abs(face_cx - prev["face_cx"]),
                abs(face_cy - prev["face_cy"]),
                abs(fw - prev["face_w"]) * 0.5,
                abs(fh - prev["face_h"]) * 0.5,
            )
            visual_stable = _face_signature_stable(prev.get("face_sig"), face_sig)
            cur_mcx = new_ax + new_aw * 0.5
            cur_mcy = new_ay + new_ah * 0.5
            mouth_delta = max(abs(cur_mcx - prev["abs_cx"]), abs(cur_mcy - prev["abs_cy"]))

            face_is_stable = visual_stable or face_delta <= stable_face_px
            if face_is_stable and mouth_delta <= mouth_deadband_px:
                stable_mcx = prev["abs_cx"] * (1.0 - stable_mouth_alpha) + cur_mcx * stable_mouth_alpha
                stable_mcy = prev["abs_cy"] * (1.0 - stable_mouth_alpha) + cur_mcy * stable_mouth_alpha
                if prev["mt"] == mt:
                    new_aw = int(round(prev["abs_w"]))
                    new_ah = int(round(prev["abs_h"]))
                new_ax = int(round(stable_mcx - new_aw * 0.5))
                new_ay = int(round(stable_mcy - new_ah * 0.5))

        nomouth_shape = store.load_nomouth(q).shape[:2]
        H_fr, W_fr = nomouth_shape
        new_ax = max(0, min(W_fr - 1, new_ax))
        new_ay = max(0, min(H_fr - 1, new_ay))
        new_aw = min(new_aw, W_fr - new_ax)
        new_ah = min(new_ah, H_fr - new_ay)

        if (new_ax, new_ay, new_aw, new_ah) != (ax, ay, aw, ah):
            store.save_mouth_frame(q, rgba, new_ax, new_ay, new_aw, new_ah)
            adjusted += 1

        prev = {
            "fg": fg, "mt": mt, "cx": cx_rel, "cy": cy_rel,
            "w": w_rel, "h": h_rel,
            "face_cx": face_cx, "face_cy": face_cy,
            "face_w": fw, "face_h": fh,
            "abs_cx": new_ax + new_aw * 0.5,
            "abs_cy": new_ay + new_ah * 0.5,
            "abs_w": new_aw, "abs_h": new_ah,
            "face_sig": face_sig,
        }

    if adjusted:
        print(f"[F2] Suavização temporal aplicada em {adjusted} frames.")


def process_single(q: int,
                   store: DiskStore,
                   mouth_types: List[int],
                   detection_model,
                   bbox_cache: Dict,
                   upscale_crop_face: float,
                   conf_mouth: float,
                   mouth_padding_per_type: Dict[int, int] = None,
                   mouth_brightness_per_type: Dict[int, float] = None,
                   use_rembg: bool = False) -> None:
    mt        = mouth_types[q]
    meta      = store.load_face_frame_meta(q)
    fg        = int(meta.get("face_group", 0))
    epoch     = int(meta.get("mouth_cache_epoch", fg))

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

    """
    if bbox_cache.get("_mouth_cache_epoch") != epoch:
        bbox_cache.clear()
        bbox_cache["_mouth_cache_epoch"] = epoch
    if store.reset_mouth_cache_epoch(epoch):
        bbox_cache.clear()
        bbox_cache["_mouth_cache_epoch"] = epoch
        print(f"  [F2] Variação de face detectada → caches closed/half/open limpos (epoch={epoch})")
    """

    cache_key = (epoch, mt, fg)
    cached = bbox_cache.get(cache_key)
    if cached is None and store.has_mouth_cache(mt, fg, epoch=epoch):
        try:
            cached_rgba, cached_meta = store.load_mouth_cache(mt, fg)
            cached = {"rgba": cached_rgba, "pos": cached_meta}
            bbox_cache[cache_key] = cached
        except Exception as exc:
            print(f"  [F2] Cache de boca inválido {MOUTH_TYPE_NAMES[mt]} fg={fg}: {exc}")

    if cached is not None:
        try:
            ax, ay, aw, ah = _mouth_cached_pos_to_frame(
                cached["pos"], nomouth_shape, face_bbox)

            store.save_mouth_frame(q, cached["rgba"], ax, ay, aw, ah)
            if q % 20 == 0:
                print(f"  [F2] Boca cache {MOUTH_TYPE_NAMES[mt]} fg={fg} frame{q} "
                      f"→ ({ax},{ay}) {aw}×{ah}px")
            return
        except Exception as exc:
            print(f"  [F2] Falha aplicando cache de boca {MOUTH_TYPE_NAMES[mt]} fg={fg}: {exc}")

    result = _detect_mouth(face_np, face_meta, face_bbox, nomouth_shape,
                           detection_model, upscale_crop_face, conf_mouth,
                           mouth_type=mt,
                           mouth_padding_per_type=mouth_padding_per_type,
                           mouth_brightness_per_type=mouth_brightness_per_type,
                           use_rembg=use_rembg)

    if result is None:
        store.save_mouth_frame(q, np.zeros((4, 4, 4), dtype=np.uint8), 0, 0, 0, 0)
        return

    store.save_mouth_frame(q, *result)
    try:
        rgba, ax, ay, aw, ah = result
        pos = {"ax": ax, "ay": ay, "aw": aw, "ah": ah}
        cache_ref_shape = face_np.shape
        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            cache_ref_shape = (
                max(1, int(fy2) - int(fy1)),
                max(1, int(fx2) - int(fx1)),
            )
            pos.update({
                "gx": ax - int(fx1),
                "gy": ay - int(fy1),
                "gw": aw,
                "gh": ah,
                "rx": (ax - int(fx1)) / max(1, cache_ref_shape[1]),
                "ry": (ay - int(fy1)) / max(1, cache_ref_shape[0]),
                "rw": aw / max(1, cache_ref_shape[1]),
                "rh": ah / max(1, cache_ref_shape[0]),
                "ref_fx1": int(fx1),
                "ref_fy1": int(fy1),
                "ref_fw": cache_ref_shape[1],
                "ref_fh": cache_ref_shape[0],
                "generated_face_w": cache_ref_shape[1],
                "generated_face_h": cache_ref_shape[0],
            })
        if use_rembg:
            pos["alpha_source"] = "easy_imageRemBg_BEN2"
        cached = {"rgba": rgba, "pos": pos}
        bbox_cache[cache_key] = cached
        if not store.has_mouth_cache(mt, fg, epoch=epoch):
            store.save_mouth_cache(
                mt, fg, rgba, pos,
                epoch=epoch,
                source_frame=q,
                generated_face_shape=cache_ref_shape,
            )
    except Exception as exc:
        print(f"  [F2] Falha salvando cache de boca {MOUTH_TYPE_NAMES[mt]} fg={fg}: {exc}")

    if q % 20 == 0:
        print(f"  [F2] Boca {MOUTH_TYPE_NAMES[mt]} fg={fg} frame{q} "
              f"→ ({result[1]},{result[2]}) {result[3]}×{result[4]}px")


def _worker_batch(worker_id: int,
                  indices: List[int],
                  store: DiskStore,
                  mouth_types: List[int],
                  detection_model_path: str,
                  upscale_crop_face: float,
                  conf_mouth: float,
                  mouth_padding_per_type: Dict[int, int] = None,
                  mouth_brightness_per_type: Dict[int, float] = None,
                  use_rembg: bool = False) -> int:
    from ultralytics import YOLO
    completed = 0
    model = None
    try:
        model = YOLO(detection_model_path)
        local: Dict = {}
        for q in indices:
            try:
                process_single(q, store, mouth_types, model, local,
                               upscale_crop_face, conf_mouth,
                               mouth_padding_per_type, mouth_brightness_per_type,
                               use_rembg=use_rembg)
            except Exception as exc:
                print(f"[F2-W{worker_id}] Erro frame {q}: {exc}")
                try:
                    store.save_mouth_frame(q, np.zeros((4, 4, 4), dtype=np.uint8), 0, 0, 0, 0)
                except Exception:
                    pass
            completed += 1
        return completed
    finally:
        try:
            del model
        except Exception:
            pass
        cuda_cleanup()


def run_phase2(store: DiskStore,
               mouth_types: List[int],
               n_frames: int,
               detection_model_path: str,
               n_workers: int,
               upscale_crop_face: float = 2.0,
               conf_mouth: float = 0.1,
               mouth_padding_per_type: Dict[int, int] = None,
               mouth_brightness_per_type: Dict[int, float] = None,
               use_rembg: bool = False) -> None:
    n_workers = max(1, n_workers)
    if use_rembg:
        print("[F2] BEN2 ativo para remover pele do recorte da boca quando disponível.")

    print(f"\n[F2] Bocas por frame estabilizadas — modelo unico face+boca, {n_frames} frames, {n_workers} workers, "
          f"upscale=auto(min={upscale_crop_face:.1f}x) conf={conf_mouth} "
          f"padding={mouth_padding_per_type} brightness={mouth_brightness_per_type} "
          f"BEN2={'on' if use_rembg else 'off'}...")

    completed = 0
    log_step = max(1, n_frames // 10)

    epoch_ranges = []
    start = 0
    cur_epoch = None
    for q in range(n_frames):
        meta = store.load_face_frame_meta(q) if store.has_face_frame(q) else {}
        epoch = int(meta.get("mouth_cache_epoch", meta.get("face_group", 0)))
        if cur_epoch is None:
            cur_epoch = epoch
        elif epoch != cur_epoch:
            epoch_ranges.append((cur_epoch, list(range(start, q))))
            start = q
            cur_epoch = epoch
    if cur_epoch is not None:
        epoch_ranges.append((cur_epoch, list(range(start, n_frames))))

    for epoch, frames in epoch_ranges:
        if store.reset_mouth_cache_epoch(epoch):
            print(f"[F2] Variação de face detectada → caches closed/half/open limpos (epoch={epoch})")

        chunks = [frames[i::n_workers] for i in range(n_workers)]
        chunks = [chunk for chunk in chunks if chunk]

        with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            futures = {
                ex.submit(_worker_batch, worker_id, chunk, store, mouth_types,
                          detection_model_path, upscale_crop_face, conf_mouth,
                          mouth_padding_per_type, mouth_brightness_per_type,
                          use_rembg): worker_id
                for worker_id, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                try:
                    completed += future.result()
                except Exception as e:
                    print(f"[F2] Worker falhou: {e}")
                if completed % log_step == 0 or completed == n_frames:
                    print(f"[F2] {completed}/{n_frames} detectados")

    #smooth_mouth_tracks(store, mouth_types, n_frames)

    cuda_cleanup()
    print(f"[F2] Concluída. {vram_info()}")
