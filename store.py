import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .constants import (
    _IMG_EXT, _WEBP_QUALITY, _MIN_WORKERS, _OOM_BACKOFF_BASE,
    MOUTH_TYPE_NAMES,
)
from .utils import jdumps, jload, atomic_write, atomic_copy


# ===========================================================================
# VramGuard
# ===========================================================================

class VramGuard:
    def __init__(self, initial_workers: int, safety_mb: int,
                 per_worker_mb: int, interval: float = 0.5):
        self._lock     = threading.Lock()
        self._cond     = threading.Condition(self._lock)
        self._max      = initial_workers
        self._active   = 0
        self._safety   = safety_mb
        self._per      = per_worker_mb
        self._interval = interval
        self._cuda     = torch.cuda.is_available()

    def _free(self) -> float:
        if not self._cuda:
            return float("inf")
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 ** 2)

    def _ok(self) -> bool:
        return self._free() >= self._safety + self._per

    def acquire(self):
        with self._cond:
            while True:
                if self._active < self._max and self._ok():
                    self._active += 1
                    return
                self._cond.wait(timeout=self._interval)

    def release(self):
        with self._cond:
            self._active = max(0, self._active - 1)
            self._cond.notify_all()

    def reduce(self, by: int = 1) -> int:
        with self._cond:
            self._max = max(_MIN_WORKERS, self._max - by)
            v = self._max
            self._cond.notify_all()
        print(f"[VramGuard] OOM → workers={v}")
        return v

    @property
    def max_concurrent(self):
        with self._lock:
            return self._max


# ===========================================================================
# DiskStore
# ===========================================================================

class DiskStore:
    PNG_GOOD = 6

    def __init__(self, work_dir: Path):
        self._mouth_cache_lock = threading.Lock()
        self._mouth_cache_epoch = 0
        self.work_dir    = work_dir
        self.orig_dir    = work_dir / "preproc" / "original"
        self.nomouth_dir = work_dir / "preproc" / "no_mouth"
        self.mmask_dir   = work_dir / "preproc" / "mouth_masks"
        self.fmask_dir   = work_dir / "preproc" / "face_masks"
        self.fbbox_dir   = work_dir / "preproc" / "face_bboxes"
        self.faces_dir    = work_dir / "diffusion" / "faces"
        self.history_dir  = work_dir / "diffusion" / "history"
        self.mouths_dir   = work_dir / "diffusion" / "mouths"
        self.mouth_cache_dir = work_dir / "diffusion" / "mouth_cache"
        self.output_dir   = work_dir / "output"
        self.charbbox_dir = work_dir / "preproc" / "character_bboxes"
        self.debug_face_frames_dir = work_dir / "debug" / "face_frames"
        self.debug_face_scribble_frames_dir = work_dir / "debug" / "face_scribble_frames"
        for d in (self.orig_dir, self.nomouth_dir, self.mmask_dir, self.fmask_dir,
                  self.fbbox_dir, self.charbbox_dir, self.faces_dir, self.history_dir,
                  self.mouths_dir, self.mouth_cache_dir, self.output_dir,
                  self.debug_face_frames_dir, self.debug_face_scribble_frames_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── paths ────────────────────────────────────────────────────────────────

    def _p(self, d: Path, idx: int, ext: str = "png") -> Path:
        return d / f"{idx:06d}.{ext}"

    def _frame_face_img(self, idx):  return self.faces_dir   / f"frame_{idx:06d}.{_IMG_EXT}"
    def _frame_face_json(self, idx): return self.faces_dir   / f"frame_{idx:06d}.json"
    def _hist_img(self, name, fg):   return self.history_dir / f"{name}_{fg}.{_IMG_EXT}"
    def _hist_json(self, name, fg):  return self.history_dir / f"{name}_{fg}.json"
    def _mouth_img(self, idx):       return self.mouths_dir  / f"frame_{idx:06d}.{_IMG_EXT}"
    def _mouth_json(self, idx):      return self.mouths_dir  / f"frame_{idx:06d}.json"
    def _mouth_cache_img(self, name, fg):  return self.mouth_cache_dir / f"{name}_{fg}.{_IMG_EXT}"
    def _mouth_cache_json(self, name, fg): return self.mouth_cache_dir / f"{name}_{fg}.json"

    # ── webp helpers ─────────────────────────────────────────────────────────

    def _save_webp_rgb(self, path, arr):
        tmp = path.with_suffix(".tmp." + _IMG_EXT)
        Image.fromarray(arr).save(tmp, format="WEBP", quality=_WEBP_QUALITY)
        tmp.replace(path)

    def _save_webp_rgba(self, path, arr):
        tmp = path.with_suffix(".tmp." + _IMG_EXT)
        Image.fromarray(arr, mode="RGBA").save(tmp, format="WEBP", lossless=True)
        tmp.replace(path)

    def _save_webp_mask(self, path, arr):
        tmp = path.with_suffix(".tmp." + _IMG_EXT)
        Image.fromarray(arr, mode="L").save(tmp, format="WEBP", lossless=True)
        tmp.replace(path)

    # ── preproc ──────────────────────────────────────────────────────────────

    def save_orig(self, idx, arr):
        Image.fromarray(arr).save(self._p(self.orig_dir, idx))

    def save_nomouth(self, idx, arr):
        Image.fromarray(arr).save(self._p(self.nomouth_dir, idx))

    def save_mmask(self, idx, arr):
        self._save_webp_mask(self._p(self.mmask_dir, idx, _IMG_EXT), arr)

    def save_fmask(self, idx, arr):
        self._save_webp_mask(self._p(self.fmask_dir, idx, _IMG_EXT), arr)

    def save_face_bbox(self, idx, x1, y1, x2, y2):
        atomic_write(self._p(self.fbbox_dir, idx, "json"),
                     jdumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2}))

    def _debug_image_array(self, arr):
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu()
            if arr.ndim == 4:
                arr = arr[0]
            arr = arr.numpy()
        else:
            arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.moveaxis(arr, 0, -1)
        if arr.dtype != np.uint8:
            max_v = float(np.nanmax(arr)) if arr.size else 0.0
            if max_v <= 1.0:
                arr = arr * 255.0
            arr = np.nan_to_num(arr).clip(0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return arr

    def save_debug_face_frame(self, idx, arr):
        arr = self._debug_image_array(arr)
        Image.fromarray(arr).save(self._p(self.debug_face_frames_dir, idx))
        
    def save_debug_face_scribble_frame(self, idx, arr):
        arr = self._debug_image_array(arr)
        Image.fromarray(arr).save(self._p(self.debug_face_scribble_frames_dir, idx))

    def save_character_bbox(self, idx, x1, y1, x2, y2):
        atomic_write(self._p(self.charbbox_dir, idx, "json"),
                     jdumps({"x1": x1, "y1": y1, "x2": x2, "y2": y2}))

    def load_character_bbox(self, idx):
        p = self._p(self.charbbox_dir, idx, "json")
        if not p.exists():
            return None
        d = jload(p)
        if not d:
            return None
        try:
            return int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        except Exception:
            return None

    def load_orig(self, idx):
        return np.array(Image.open(self._p(self.orig_dir, idx)).convert("RGB"), dtype=np.uint8)

    def load_nomouth(self, idx):
        return np.array(Image.open(self._p(self.nomouth_dir, idx)).convert("RGB"), dtype=np.uint8)

    def load_mmask(self, idx):
        for ext in (_IMG_EXT, "png"):
            p = self._p(self.mmask_dir, idx, ext)
            if p.exists():
                return np.array(Image.open(p).convert("L"), dtype=np.uint8)
        return np.zeros(self.load_nomouth(idx).shape[:2], dtype=np.uint8)

    def load_fmask(self, idx):
        for ext in (_IMG_EXT, "png"):
            p = self._p(self.fmask_dir, idx, ext)
            if p.exists():
                return np.array(Image.open(p).convert("L"), dtype=np.uint8)
        ref = self._p(self.nomouth_dir, idx)
        if ref.exists():
            pil = Image.open(ref)
            return np.ones((pil.size[1], pil.size[0]), dtype=np.uint8) * 255
        return np.ones((512, 512), dtype=np.uint8) * 255

    def load_face_bbox(self, idx) -> Optional[Tuple[int, int, int, int]]:
        p = self._p(self.fbbox_dir, idx, "json")
        if not p.exists():
            return None
        d = jload(p)
        if not d:
            return None
        try:
            return int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
        except Exception:
            return None

    # ── diffusion / faces ─────────────────────────────────────────────────────

    def has_face_frame(self, idx):
        return self._frame_face_img(idx).exists()

    def load_face_frame(self, idx):
        return np.array(Image.open(self._frame_face_img(idx)).convert("RGB"), dtype=np.uint8)

    def load_face_frame_meta(self, idx):
        return jload(self._frame_face_json(idx)) or {}

    def save_face_frame(self, idx, arr, mouth_type, face_group, prompt, source,
                        crop_w=0, crop_h=0, mouth_cache_epoch=0,
                        use_mouth_cache=True):
        self._save_webp_rgb(self._frame_face_img(idx), arr)
        atomic_write(self._frame_face_json(idx), jdumps({
            "mouth_type": mouth_type, "mouth_type_name": MOUTH_TYPE_NAMES[mouth_type],
            "face_group": face_group, "prompt": prompt, "source": source,
            "w": arr.shape[1], "h": arr.shape[0], "crop_w": crop_w, "crop_h": crop_h,
            "mouth_cache_epoch": mouth_cache_epoch,
            "use_mouth_cache": bool(use_mouth_cache),
        }))

    def copy_face_to_frame(self, src_img, idx, mouth_type, face_group, prompt,
                           mouth_cache_epoch=0):
        atomic_copy(src_img, self._frame_face_img(idx))
        d = jload(src_img.with_suffix(".json")) or {}
        atomic_write(self._frame_face_json(idx), jdumps({
            "mouth_type": mouth_type, "mouth_type_name": MOUTH_TYPE_NAMES[mouth_type],
            "face_group": face_group, "prompt": prompt, "source": "history",
            "w": d.get("w", 0), "h": d.get("h", 0),
            "crop_w": d.get("crop_w", 0), "crop_h": d.get("crop_h", 0),
            "mouth_cache_epoch": mouth_cache_epoch,
        }))

    # ── history ───────────────────────────────────────────────────────────────

    def has_history(self, mt, fg):
        return self._hist_img(MOUTH_TYPE_NAMES[mt], fg).exists()

    def load_history_img_path(self, mt, fg):
        return self._hist_img(MOUTH_TYPE_NAMES[mt], fg)

    def save_history(self, mt, fg, arr, prompt, crop_w=0, crop_h=0):
        name = MOUTH_TYPE_NAMES[mt]
        self._save_webp_rgb(self._hist_img(name, fg), arr)
        atomic_write(self._hist_json(name, fg), jdumps({
            "mouth_type": mt, "mouth_type_name": name, "face_group": fg,
            "prompt": prompt, "w": arr.shape[1], "h": arr.shape[0],
            "crop_w": crop_w, "crop_h": crop_h,
        }))

    # ── mouths ────────────────────────────────────────────────────────────────

    def has_mouth_frame(self, idx):
        return self._mouth_img(idx).exists() and self._mouth_json(idx).exists()

    def save_mouth_frame(self, idx, rgba, ax, ay, aw, ah):
        self._save_webp_rgba(self._mouth_img(idx), rgba)
        atomic_write(self._mouth_json(idx),
                     jdumps({"ax": ax, "ay": ay, "aw": aw, "ah": ah, "frame": idx}))

    def load_mouth_frame(self, idx):
        rgba = np.array(Image.open(self._mouth_img(idx)).convert("RGBA"), dtype=np.uint8)
        d    = jload(self._mouth_json(idx)) or {}
        return rgba, int(d.get("ax", 0)), int(d.get("ay", 0)), \
               int(d.get("aw", 0)), int(d.get("ah", 0))

    def clear_mouth_cache(self):
        removed = 0
        with self._mouth_cache_lock:
            for p in self.mouth_cache_dir.iterdir():
                if p.is_file():
                    try:
                        p.unlink()
                        removed += 1
                    except OSError:
                        pass
        return removed

    def _mouth_cache_meta_valid(self, meta: dict, epoch=None) -> bool:
        required = ("ax", "ay", "aw", "ah")
        if not all(k in meta for k in required):
            return False
        try:
            if int(meta.get("position_schema", 0)) != 3:
                return False
            if int(meta.get("aw", 0)) <= 0 or int(meta.get("ah", 0)) <= 0:
                return False
            if any(k in meta for k in ("gx", "gy", "gw", "gh")):
                if not all(k in meta for k in ("gx", "gy", "gw", "gh")):
                    return False
                if int(meta.get("gw", 0)) <= 0 or int(meta.get("gh", 0)) <= 0:
                    return False
            if any(k in meta for k in ("ref_fx1", "ref_fy1", "ref_fw", "ref_fh")):
                if not all(k in meta for k in ("ref_fx1", "ref_fy1", "ref_fw", "ref_fh")):
                    return False
                if int(meta.get("ref_fw", 0)) <= 0 or int(meta.get("ref_fh", 0)) <= 0:
                    return False
            if epoch is not None and int(meta.get("mouth_cache_epoch", -1)) != int(epoch):
                return False
        except Exception:
            return False
        return True

    def reset_mouth_cache_epoch(self, epoch: int) -> bool:
        with self._mouth_cache_lock:
            cur = getattr(self, "_mouth_cache_epoch", None)
            if cur == epoch:
                return False
            self._mouth_cache_epoch = epoch
        self.clear_mouth_cache()
        return True

    def has_mouth_cache(self, mt, fg, epoch=None):
        name = MOUTH_TYPE_NAMES[mt]
        img_path = self._mouth_cache_img(name, fg)
        json_path = self._mouth_cache_json(name, fg)
        if not img_path.exists() or not json_path.exists():
            return False
        meta = jload(json_path) or {}
        return self._mouth_cache_meta_valid(meta, epoch=epoch)

    def save_mouth_cache(self, mt, fg, rgba, pos, epoch=0,
                         source_frame=None, generated_face_shape=None):
        name = MOUTH_TYPE_NAMES[mt]
        face_h = face_w = 0
        if generated_face_shape is not None:
            face_h, face_w = generated_face_shape[:2]
        with self._mouth_cache_lock:
            self._save_webp_rgba(self._mouth_cache_img(name, fg), rgba)
            meta = {
                "mouth_type": mt,
                "mouth_type_name": name,
                "face_group": fg,
                "mouth_cache_epoch": epoch,
                "position_source": "generated_face",
                "position_schema": 3,
                "source_frame": source_frame,
                "generated_face_w": face_w,
                "generated_face_h": face_h,
                "ax": pos["ax"],
                "ay": pos["ay"],
                "aw": pos["aw"],
                "ah": pos["ah"],
                "rgba_w": rgba.shape[1],
                "rgba_h": rgba.shape[0],
            }
            if all(k in pos for k in ("gx", "gy", "gw", "gh")):
                meta.update({
                    "gx": pos["gx"],
                    "gy": pos["gy"],
                    "gw": pos["gw"],
                    "gh": pos["gh"],
                })
            if all(k in pos for k in ("ref_fx1", "ref_fy1", "ref_fw", "ref_fh")):
                meta.update({
                    "ref_fx1": pos["ref_fx1"],
                    "ref_fy1": pos["ref_fy1"],
                    "ref_fw": pos["ref_fw"],
                    "ref_fh": pos["ref_fh"],
                })
            atomic_write(self._mouth_cache_json(name, fg), jdumps(meta))

    def load_mouth_cache(self, mt, fg):
        name = MOUTH_TYPE_NAMES[mt]
        rgba = np.array(Image.open(self._mouth_cache_img(name, fg)).convert("RGBA"), dtype=np.uint8)
        meta = jload(self._mouth_cache_json(name, fg)) or {}
        if not self._mouth_cache_meta_valid(meta):
            raise ValueError(f"cache de boca sem posição absoluta válida: {name}_{fg}")
        return rgba, meta

    # ── output ────────────────────────────────────────────────────────────────

    def has_output(self, idx):
        return self._p(self.output_dir, idx).exists()

    def save_output(self, idx, arr):
        Image.fromarray(arr).save(
            self._p(self.output_dir, idx), format="PNG", compress_level=self.PNG_GOOD)

    def load_output(self, idx):
        return np.array(Image.open(self._p(self.output_dir, idx)).convert("RGB"), dtype=np.uint8)
