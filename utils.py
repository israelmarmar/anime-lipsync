import gc
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import soundfile as sf
import torch
from PIL import Image

from .constants import _MIN_WORKERS, _MAX_WORKERS, _VRAM_PER_WORKER_MB


# ===========================================================================
# JSON
# ===========================================================================

class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def jdumps(d: dict) -> str:
    return json.dumps(d, cls=_NpEncoder)


def jload(path: Path) -> Optional[dict]:
    try:
        t = path.read_text(encoding="utf-8").strip()
        return json.loads(t) if t else None
    except (json.JSONDecodeError, OSError):
        return None


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp" + dst.suffix)
    shutil.copy2(src, tmp)
    tmp.replace(dst)


# ===========================================================================
# VRAM
# ===========================================================================

def free_vram(*tensors) -> None:
    for t in tensors:
        if t is not None:
            del t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def cuda_cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def vram_free_mb() -> float:
    if not torch.cuda.is_available():
        return float("inf")
    free, _ = torch.cuda.mem_get_info()
    return free / (1024 ** 2)


def vram_info() -> str:
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e6
        rsv  = torch.cuda.memory_reserved()  / 1e6
        return f"VRAM: {used:.0f}MB alloc / {rsv:.0f}MB reserv"
    return "CUDA indisponível"


def compute_workers(safety_mb: int = 1024,
                    per_mb: int = _VRAM_PER_WORKER_MB) -> int:
    if not torch.cuda.is_available():
        return max(_MIN_WORKERS, min((os.cpu_count() or 2) // 2, _MAX_WORKERS))
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_mb  = free_bytes  / (1024 ** 2)
    total_mb = total_bytes / (1024 ** 2)
    usable   = max(0.0, free_mb - safety_mb)
    w        = max(_MIN_WORKERS, min(int(usable // per_mb), _MAX_WORKERS))
    print(f"[Workers] VRAM total:{total_mb:,.0f}MB livre:{free_mb:,.0f}MB → {w} workers")
    return w


# ===========================================================================
# Tensor ↔ numpy
# ===========================================================================

def t2np(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    return (t.numpy() * 255).clip(0, 255).astype(np.uint8)


def np2t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(0)


def get_node_output(obj, index: int):
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


# ===========================================================================
# Áudio
# ===========================================================================

def comfy_audio_to_wav(audio: dict, tmp_dir: str) -> str:
    waveform    = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.dim() == 3: waveform = waveform.squeeze(0)
    if waveform.dim() == 2: waveform = waveform.mean(dim=0)
    wav_np = waveform.cpu().numpy().astype(np.float32)
    if sample_rate != 16000:
        import librosa as _lb
        wav_np      = _lb.resample(wav_np, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    path = os.path.join(tmp_dir, "audio_input.wav")
    sf.write(path, wav_np, sample_rate, subtype="PCM_16")
    return path


def save_audio_original(audio: dict, tmp_dir: str) -> str:
    waveform    = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.dim() == 3: waveform = waveform.squeeze(0)
    if waveform.dim() == 2: waveform = waveform.mean(dim=0)
    wav_np = waveform.cpu().numpy().astype(np.float32)
    path   = os.path.join(tmp_dir, "audio_original.wav")
    sf.write(path, wav_np, sample_rate, subtype="PCM_16")
    return path


# ===========================================================================
# ComfyUI helpers
# ===========================================================================

def get_comfy_output_dir() -> str:
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except Exception:
        fallback = "/tmp/comfyui_lipsync_output"
        os.makedirs(fallback, exist_ok=True)
        return fallback


def unique_filename(output_dir: str, stem: str, ext: str = ".mp4") -> Tuple[str, str]:
    counter = 1
    name = f"{stem}{ext}"
    while os.path.exists(os.path.join(output_dir, name)):
        name = f"{stem}_{counter:04d}{ext}"
        counter += 1
    return os.path.join(output_dir, name), name


# ===========================================================================
# ffmpeg
# ===========================================================================

def build_video_with_audio(output_dir: Path, audio_path: str,
                            n_frames: int, fps: int, out_path: str) -> str:
    frame_pattern = str(output_dir / "%06d.png")
    tmp_video     = out_path.replace(".mp4", "_noaudio.mp4")

    r = subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps), "-i", frame_pattern,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p", tmp_video,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg (vídeo):\n{r.stderr}")

    r = subprocess.run([
        "ffmpeg", "-y", "-i", tmp_video, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", out_path,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg (mux):\n{r.stderr}")

    try:
        os.remove(tmp_video)
    except OSError:
        pass
    return out_path


# ===========================================================================
# Máscara / inpainting
# ===========================================================================

def build_mask(H: int, W: int, bbox, dilation: int, blur: int,
               base_mask: np.ndarray = None) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    if base_mask is not None and np.max(base_mask) > 0:
        if base_mask.shape[:2] != (H, W):
            base_mask = cv2.resize(base_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = (base_mask > 127).astype(np.uint8) * 255
    elif bbox is not None:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
    else:
        return mask
    if dilation > 0:
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation+1, 2*dilation+1))
        mask = cv2.dilate(mask, k, iterations=1)
    if blur > 0:
        ks   = blur * 2 + 1
        mask = cv2.GaussianBlur(mask, (ks, ks), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def cv2_inpaint(img: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    if mask.max() == 0:
        return img.copy()
    return cv2.inpaint(img, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
