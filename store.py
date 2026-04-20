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
        self.work_dir    = work_dir
        self.orig_dir    = work_dir / "preproc" / "original"
        self.nomouth_dir = work_dir / "preproc" / "no_mouth"
        self.mmask_dir   = work_dir / "preproc" / "mouth_masks"
        self.fmask_dir   = work_dir / "preproc" / "face_masks"
        self.fbbox_dir   = work_dir / "preproc" / "face_bboxes"
        self.faces_dir    = work_dir / "diffusion" / "faces"
        self.history_dir  = work_dir / "diffusion" / "history"
        self.mouths_dir   = work_dir / "diffusion" / "mouths"
        self.output_dir   = work_dir / "output"
        self.charbbox_dir = work_dir / "preproc" / "character_bboxes"
        for d in (self.orig_dir, self.nomouth_dir, self.mmask_dir, self.fmask_dir,
                  self.fbbox_dir, self.charbbox_dir, self.faces_dir, self.history_dir,
                  self.mouths_dir, self.output_dir):
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
                        crop_w=0, crop_h=0):
        self._save_webp_rgb(self._frame_face_img(idx), arr)
        atomic_write(self._frame_face_json(idx), jdumps({
            "mouth_type": mouth_type, "mouth_type_name": MOUTH_TYPE_NAMES[mouth_type],
            "face_group": face_group, "prompt": prompt, "source": source,
            "w": arr.shape[1], "h": arr.shape[0], "crop_w": crop_w, "crop_h": crop_h,
        }))

    def copy_face_to_frame(self, src_img, idx, mouth_type, face_group, prompt):
        atomic_copy(src_img, self._frame_face_img(idx))
        d = jload(src_img.with_suffix(".json")) or {}
        atomic_write(self._frame_face_json(idx), jdumps({
            "mouth_type": mouth_type, "mouth_type_name": MOUTH_TYPE_NAMES[mouth_type],
            "face_group": face_group, "prompt": prompt, "source": "history",
            "w": d.get("w", 0), "h": d.get("h", 0),
            "crop_w": d.get("crop_w", 0), "crop_h": d.get("crop_h", 0),
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

    # ── output ────────────────────────────────────────────────────────────────

    def has_output(self, idx):
        return self._p(self.output_dir, idx).exists()

    def save_output(self, idx, arr):
        Image.fromarray(arr).save(
            self._p(self.output_dir, idx), format="PNG", compress_level=self.PNG_GOOD)

    def load_output(self, idx):
        return np.array(Image.open(self._p(self.output_dir, idx)).convert("RGB"), dtype=np.uint8)
