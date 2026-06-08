import gc
import math
import random
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
import torch
from PIL import Image

from .constants import (
    MOUTH_TYPE_NAMES, MOUTH_TYPE_PROMPTS,
    _VRAM_PER_KSAMPLER_MB, _QUEUE_SENTINEL,
)
from .lora import apply_lora
from .store import DiskStore
from .utils import (
    t2np, np2t, get_node_output, free_vram, cuda_cleanup,
    vram_free_mb, vram_info,
)

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None

_HED_AUX_DETECTOR = None
_HED_AUX_UNAVAILABLE_REASON = None
_HED_GROUP_WARNED = False


def _resize_for_similarity(img, target_w: int = 256):
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return img
    scale = target_w / max(w, 1)
    target_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _face_only_canvas(rgb, face_bbox):
    import numpy as np

    if not face_bbox:
        return rgb

    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = face_bbox
    x1 = max(0, min(w, int(x1)))
    x2 = max(0, min(w, int(x2)))
    y1 = max(0, min(h, int(y1)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return rgb

    canvas = np.zeros_like(rgb)
    canvas[y1:y2, x1:x2] = rgb[y1:y2, x1:x2]
    return canvas


def _cv2_face_scribble_from_norm(norm) -> Tuple:
    import numpy as np

    med = float(np.median(norm))
    lo = int(max(20, 0.66 * med))
    hi = int(min(180, 1.33 * med + 30))
    canny = cv2.Canny(norm, lo, hi)

    block = 15 if min(norm.shape[:2]) >= 64 else 9
    adaptive = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, block, 5)

    scribble = cv2.bitwise_or(canny, adaptive)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.morphologyEx(scribble, cv2.MORPH_OPEN, k, iterations=1)


def _preprocessor_face_scribble(rgb, preprocessor, target_shape) -> Tuple:
    hed_out = preprocessor.execute(resolution=512, 
                    image=np2t(rgb), safe="enable")
    return get_node_output(hed_out,0)


def _face_crop_feature(rgb) -> np.ndarray:
    small = _resize_for_similarity(rgb)
    if small.ndim == 3:
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    else:
        gray = small
    if gray.shape[0] > 5 and gray.shape[1] > 5:
        gray = cv2.bilateralFilter(gray, 5, 45, 45)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _scribble_to_uint8_array(scribble, target_shape) -> np.ndarray:
    if isinstance(scribble, torch.Tensor):
        arr = scribble.detach().cpu()
        if arr.ndim == 4:
            arr = arr[0]
        arr = arr.numpy()
    elif isinstance(scribble, Image.Image):
        arr = np.asarray(scribble)
    else:
        arr = np.asarray(scribble)

    if arr.dtype != np.uint8:
        max_v = float(np.nanmax(arr)) if arr.size else 0.0
        if max_v <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr).clip(0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)

    target_h, target_w = target_shape
    if arr.shape[:2] != (target_h, target_w):
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return arr


def _face_scribble(rgb, preprocessor) -> Tuple:
    """
    Extrai um scribble da face para comparar variações de linha/pose.

    Anime costuma mudar pouco em textura e muito nos contornos: olhos, boca,
    mandíbula, cabelo sobre o rosto. Quando disponível, usa o mesmo
    Preprocessor do ControlNet para agrupar por linhas mais consistentes.
    """
    global _HED_GROUP_WARNED

    norm = _face_crop_feature(rgb)

    try:
        scribble = _preprocessor_face_scribble(rgb, preprocessor, norm.shape[:2])
        scribble = _scribble_to_uint8_array(scribble, norm.shape[:2])
    except Exception as exc:
        if not _HED_GROUP_WARNED:
            print(f"  [F1] HED/scribble do agrupamento falhou ({exc}); usando CV2.")
            _HED_GROUP_WARNED = True
        scribble = _cv2_face_scribble_from_norm(norm)
    return norm, scribble


def _diff_similarity(a, b) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    if a.size == 0 or b.size == 0:
        return 0.0
    return 1.0 - float(np.mean(cv2.absdiff(a, b)) / 255.0)


def _safe_ssim(a, b):
    if ssim is None or a.size == 0 or b.size == 0:
        return None
    min_side = min(a.shape[0], a.shape[1], b.shape[0], b.shape[1])
    if min_side < 3:
        return None
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size -= 1
    try:
        return float(ssim(a, b, full=True, win_size=win_size)[0])
    except ValueError:
        return None


def _face_similarity(gray_a, gray_b) -> float:
    if gray_a.shape != gray_b.shape:
        import cv2
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
    d = _diff_similarity(gray_a, gray_b)
    s = _safe_ssim(gray_a, gray_b)
    if s is None:
        return max(0., min(1., d))
    return max(0., min(1., s * 0.7 + d * 0.3))


def _scribble_similarity(prev_feat, feat) -> float:
    import numpy as np

    gray_a, scribble_a = prev_feat
    gray_b, scribble_b = feat
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
        scribble_b = cv2.resize(scribble_b, (scribble_a.shape[1], scribble_a.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

    gray_sim = _face_similarity(gray_a, gray_b)

    scribble_ssim = _safe_ssim(scribble_a, scribble_b)
    if scribble_ssim is None:
        scribble_ssim = _diff_similarity(scribble_a, scribble_b)

    a = scribble_a > 0
    b = scribble_b > 0
    inter = float(np.logical_and(a, b).sum())
    denom = float(a.sum() + b.sum())
    edge_f1 = (2.0 * inter / denom) if denom > 0 else 1.0

    scribble_sim = max(0.0, min(1.0, scribble_ssim * 0.45 + edge_f1 * 0.55))
    return max(0.0, min(1.0, scribble_sim * 0.70 + gray_sim * 0.30))


def _get_controlnet_aux_hed():
    global _HED_AUX_DETECTOR, _HED_AUX_UNAVAILABLE_REASON

    if _HED_AUX_DETECTOR is not None:
        return _HED_AUX_DETECTOR
    if _HED_AUX_UNAVAILABLE_REASON is not None:
        raise RuntimeError(_HED_AUX_UNAVAILABLE_REASON)

    try:
        from controlnet_aux import HEDdetector

        detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
        if torch.cuda.is_available() and hasattr(detector, "to"):
            detector = detector.to("cuda")
        _HED_AUX_DETECTOR = detector
        return _HED_AUX_DETECTOR
    except Exception as exc:
        _HED_AUX_UNAVAILABLE_REASON = str(exc)
        raise RuntimeError(_HED_AUX_UNAVAILABLE_REASON) from exc


def _hed_aux_scribble_tensor(rgb, resolution: int = 512):
    detector = _get_controlnet_aux_hed()
    image = Image.fromarray(rgb)
    kwargs = {
        "detect_resolution": int(resolution),
        "image_resolution": int(resolution),
        "scribble": True,
    }
    try:
        detected = detector(image, **kwargs)
    except TypeError:
        kwargs.pop("image_resolution", None)
        detected = detector(image, **kwargs)

    import numpy as np

    if isinstance(detected, Image.Image):
        arr = np.asarray(detected.convert("RGB"))
    else:
        arr = np.asarray(detected)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.shape[-1] == 4:
            arr = arr[..., :3]
    target_h, target_w = rgb.shape[:2]
    if arr.shape[0] != target_h or arr.shape[1] != target_w:
        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return np2t(arr.astype(np.uint8))


def _make_control_scribble(face_rgb, face_t, preprocessor, mode: str = "controlnet_aux"):
    mode = (mode or "controlnet_aux").strip().lower()
    if mode in {"controlnet_aux", "auto"}:
        try:
            hed_t = _hed_aux_scribble_tensor(face_rgb, resolution=512)
            return hed_t, "controlnet_aux.HEDdetector(scribble)"
        except Exception as exc:
            if mode == "controlnet_aux":
                raise RuntimeError(
                    "controlnet_aux.HEDdetector falhou. Instale/prepare controlnet_aux "
                    "ou use hed_detector_mode='comfy_hed'. Erro: "
                    f"{exc}"
                ) from exc
            print(f"  [HED] controlnet_aux indisponível ({exc}); usando Preprocessor.")

    hed_out = preprocessor.execute(safe="enable", image=face_t, resolution=512)
    hed_t = get_node_output(hed_out, 0)
    del hed_out
    return hed_t, "ComfyUI Preprocessor"


def run_phase1(store: DiskStore,
               phonemes: List[str],
               mouth_types: List[int],
               n_frames: int,
               cfg_per_type: Dict[int, dict],
               model_obj,
               model_patch_obj,
               clip_obj,
               vae_obj,
               pos_conds_ext: Optional[Dict[int, Any]] = None,
               sim_threshold: float = 0.92,
               vram_safety_mb: int = 1024,
               ready_queue: Optional[Queue] = None,
               upscale_crop_face: float = 1.0,
               lora_cfg: Optional[Dict[int, dict]] = None,
               hed_detector_mode: str = "auto",
               ) -> None:
    """
    lora_cfg formato:
      { mt: {"path": str, "strength_model": float, "strength_clip": float} }
    """
    from nodes import VAEEncode, VAEDecode, KSampler, CLIPTextEncode, NODE_CLASS_MAPPINGS

    lora_cfg = lora_cfg or {}
    print(f"\n[F1] Preparando nós Z-Turbo... {vram_info()}")
    print(f"[F1] HED/scribble mode: {hed_detector_mode}")

    cannypreprocessor    = NODE_CLASS_MAPPINGS["CannyEdgePreprocessor"]()
    preprocessor         = NODE_CLASS_MAPPINGS["HEDPreprocessor"]()
    qwenimagediffsynthcn = NODE_CLASS_MAPPINGS["QwenImageDiffsynthControlnet"]()
    vaeencode            = VAEEncode()
    vaedecode            = VAEDecode()
    ksampler             = KSampler()
    cliptextencode       = CLIPTextEncode()

    # ── Condicionamentos + LoRA por tipo ─────────────────────────────────────
    print("[F1] Preparando condicionamentos e LoRAs...")
    _ext = pos_conds_ext or {}

    model_per_type: Dict[int, Any] = {}
    clip_per_type:  Dict[int, Any] = {}
    pos_conds:      Dict[int, Any] = {}

    for mt in range(3):
        lcfg      = lora_cfg.get(mt, {})
        lora_path = lcfg.get("path", "").strip()

        if lora_path:
            m_lora, c_lora = apply_lora(
                lora_path, model_obj, clip_obj,
                lcfg.get("strength_model", 1.0),
                lcfg.get("strength_clip",  1.0),
            )
        else:
            m_lora, c_lora = model_obj, clip_obj

        model_per_type[mt] = m_lora
        clip_per_type[mt]  = c_lora

        if mt in _ext:
            pos_conds[mt] = _ext[mt]
            print(f"  [{MOUTH_TYPE_NAMES[mt]}] CONDITIONING externo"
                  + (f" + LoRA={lora_path}" if lora_path else ""))
        else:
            enc = cliptextencode.encode(text=MOUTH_TYPE_PROMPTS[mt], clip=c_lora)
            pos_conds[mt] = get_node_output(enc, 0)
            del enc
            print(f"  [{MOUTH_TYPE_NAMES[mt]}] prompt interno"
                  + (f" + LoRA={lora_path}" if lora_path else " (sem LoRA)"))

    neg_enc  = cliptextencode.encode(text="", clip=clip_obj)
    neg_cond = get_node_output(neg_enc, 0)
    del neg_enc
    gc.collect()
    print(f"[F1] Condicionamentos prontos. {vram_info()}")

    # ── Passo 1: face_groups ─────────────────────────────────────────────────
    import cv2 as _cv2

    print(f"\n[F1] Passo 1/4: face_groups por crop da face sem boca ({n_frames} frames)...")
    print(f"  [Debug] face_frames: {store.debug_face_frames_dir}")
    face_group = 0
    last_feat  = None
    frame_meta: List[Tuple[int, int, int, int]] = []

    for q in range(n_frames):
        mt        = mouth_types[q]
        face_bbox = store.load_face_bbox(q)
        orig      = store.load_orig(q)
        nomouth   = store.load_nomouth(q)
        face_frame = _face_only_canvas(orig, face_bbox)
        store.save_debug_face_frame(q, face_frame)
        
        crop = (nomouth[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
                if face_bbox else nomouth)

        feat = _face_crop_feature(crop)
        store.save_debug_face_frame(q, crop)
        
        if last_feat is not None:
            sim = _face_similarity(last_feat, feat)
            if sim <= sim_threshold:
                face_group += 1
                removed = store.clear_mouth_cache()
                print(f"  [F1] Crop da face sem boca mudou frame {q} "
                      f"(sim={sim:.3f}) → group={face_group}; "
                      f"caches closed/half/open limpos ({removed} arquivos)")
        last_feat = feat
        frame_meta.append((q, mt, face_group, face_group))
        del face_frame, orig, nomouth

    # ── Passo 2: pares únicos ────────────────────────────────────────────────
    print("\n[F1] Passo 2/4: pares únicos...")

    pairs_to_gen: Dict[Tuple[int, int], int] = {}
    for q, mt, fg, _epoch in frame_meta:
        key = (mt, fg)
        if not store.has_history(mt, fg) and key not in pairs_to_gen:
            pairs_to_gen[key] = q

    hits = sum(1 for _, mt, fg, _epoch in frame_meta if (mt, fg) not in pairs_to_gen)
    print(f"  → {len(pairs_to_gen)} diffusions | {hits} cache hits de {n_frames}")

    # ── Passo 3: HED + KSampler + VAEDecode ─────────────────────────────────
    if pairs_to_gen:
        print(f"\n[F1] Passo 3/4: gerando {len(pairs_to_gen)} faces...")

        precomputed: Dict[Tuple[int, int], dict] = {}
        for (mt, fg), q_repr in pairs_to_gen.items():
            face_bbox = store.load_face_bbox(q_repr)
            nomouth   = store.load_nomouth(q_repr)
            face_inp  = (nomouth[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
                         if face_bbox else nomouth)
            crop_h, crop_w = face_inp.shape[:2]

            # Largura fixa de 512px para diffusion — mantém aspect ratio.
            # Garante resolução consistente independente do tamanho original do crop.
            target_w = 512
            scale    = target_w / max(crop_w, 1)
            up_W     = target_w
            up_H     = max(1, int(round(crop_h * scale)))
            if up_W != crop_w or up_H != crop_h:
                face_for_diff = _cv2.resize(face_inp, (up_W, up_H),
                                            interpolation=_cv2.INTER_LANCZOS4)
                print(f"  [HED] {MOUTH_TYPE_NAMES[mt]} group={fg} repr={q_repr} "
                      f"crop {crop_w}×{crop_h} → {up_W}×{up_H} (W=512 fixo)")
            else:
                face_for_diff = face_inp
                print(f"  [HED] {MOUTH_TYPE_NAMES[mt]} group={fg} repr={q_repr} "
                      f"crop {crop_w}×{crop_h} (já 512px)")

            face_t = np2t(face_for_diff)
            dcfg = cfg_per_type[mt]

            hed_t, hed_source = _make_control_scribble(
                face_for_diff, face_t, preprocessor,
                mode=hed_detector_mode,
            )
            print(f"  [HED] controle: {hed_source}")
            del face_for_diff

            cn_out  = qwenimagediffsynthcn.diffsynth_controlnet(
                strength=float(dcfg.get("controlnet_strength", 0.6)),
                model=model_per_type[mt],
                model_patch=model_patch_obj,
                vae=vae_obj, image=hed_t)
            patched = get_node_output(cn_out, 0)
            del cn_out, hed_t

            lat_enc   = vaeencode.encode(pixels=face_t, vae=vae_obj)
            latent_in = get_node_output(lat_enc, 0)
            del lat_enc, face_t

            precomputed[(mt, fg)] = dict(
                patched=patched, latent_in=latent_in,
                mt=mt, fg=fg, q_repr=q_repr,
                crop_w=crop_w, crop_h=crop_h,
            )
            del nomouth

        cuda_cleanup()
        print(f"  [HED] Todos prontos. {vram_info()}")

        free_mb   = vram_free_mb()
        usable_mb = max(0., free_mb - vram_safety_mb)
        block_sz  = max(1, min(len(precomputed), int(usable_mb // _VRAM_PER_KSAMPLER_MB)))
        print(f"  [KSampler] VRAM livre={free_mb:.0f}MB → bloco={block_sz}")

        pairs_list = list(precomputed.items())
        n_pairs    = len(pairs_list)

        for blk_start in range(0, n_pairs, block_sz):
            blk = pairs_list[blk_start: blk_start + block_sz]
            print(f"\n  [KSampler] Bloco {blk_start // block_sz + 1}/"
                  f"{math.ceil(n_pairs / block_sz)} ({len(blk)} pares) | {vram_info()}")

            latents_out = []
            for (mt, fg), pc in blk:
                dcfg = cfg_per_type[mt]
                try:
                    sampled = ksampler.sample(
                        seed=random.randint(1, 2**64),
                        steps=int(dcfg.get("steps", 4)),
                        cfg=float(dcfg["cfg"]),
                        sampler_name=dcfg.get("sampler", "res_multistep"),
                        scheduler=dcfg.get("scheduler", "simple"),
                        denoise=float(dcfg["denoise"]),
                        model=pc["patched"],
                        positive=pos_conds[mt],
                        negative=neg_cond,
                        latent_image=pc["latent_in"],
                    )
                    latent = get_node_output(sampled, 0)
                    del sampled
                    latents_out.append((mt, fg, pc, latent))
                except Exception as exc:
                    print(f"  [KSampler] ERRO {MOUTH_TYPE_NAMES[mt]} group={fg}: {exc}")
                    latents_out.append((mt, fg, pc, None))

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for mt, fg, pc, latent in latents_out:
                if latent is None:
                    free_vram(pc["patched"], pc["latent_in"])
                    continue
                try:
                    decoded = vaedecode.decode(samples=latent, vae=vae_obj)
                    result  = t2np(get_node_output(decoded, 0))
                    del decoded, latent
                    free_vram(pc["patched"], pc["latent_in"])

                    # Resize para crop_w×crop_h → remap 1:1 na Fase 2
                    target_w, target_h = pc["crop_w"], pc["crop_h"]
                    if (result.shape[1] != target_w or result.shape[0] != target_h) \
                            and target_w > 0 and target_h > 0:
                        result = _cv2.resize(result, (target_w, target_h),
                                             interpolation=_cv2.INTER_LANCZOS4)

                    store.save_history(mt, fg, result, MOUTH_TYPE_PROMPTS[mt],
                                       crop_w=target_w, crop_h=target_h)
                    del result
                    print(f"  [Saved] {MOUTH_TYPE_NAMES[mt]} group={fg}")
                except Exception as exc:
                    print(f"  [VAEDecode] ERRO {MOUTH_TYPE_NAMES[mt]} group={fg}: {exc}")
                    free_vram(pc["patched"], pc["latent_in"])

            cuda_cleanup()

        for pc in precomputed.values():
            try:
                free_vram(pc.get("patched"), pc.get("latent_in"))
            except Exception:
                pass
        del precomputed

    else:
        print("\n[F1] Passo 3/4: todos os pares já no histórico.")

    # ── Passo 4: distribui frames ────────────────────────────────────────────
    print(f"\n[F1] Passo 4/4: distribuindo {n_frames} frames...")


    for q, mt, fg, epoch in frame_meta:
        if store.has_face_frame(q):
            if ready_queue is not None:
                ready_queue.put(q)
            continue

        store.copy_face_to_frame(
            store.load_history_img_path(mt, fg), q, mt, fg, MOUTH_TYPE_PROMPTS[mt],
            mouth_cache_epoch=epoch)
        if ready_queue is not None:
            ready_queue.put(q)
        if (q + 1) % max(1, n_frames // 10) == 0 or q == n_frames - 1:
            print(f"  [F1] {q + 1}/{n_frames} distribuídos")

    if ready_queue is not None:
        ready_queue.put(_QUEUE_SENTINEL)
        print("[F1] Sentinel enviado.")

    # Libera model patched dos tipos que tiveram LoRA
    for mt, m in model_per_type.items():
        if m is not model_obj:
            try:
                free_vram(m)
            except Exception:
                pass

    free_vram(neg_cond, *pos_conds.values())
    cuda_cleanup()
    print(f"[F1] Concluído. {vram_info()}")
