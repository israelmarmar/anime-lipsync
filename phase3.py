import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Empty, Queue
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from .constants import _QUEUE_SENTINEL, _MAX_WORKERS, _VRAM_PER_WORKER_MB
from .store import DiskStore
from .phase2 import process_single
from .utils import cuda_cleanup, vram_free_mb, vram_info


# ===========================================================================
# Composição sem borda
# ===========================================================================

def composite(base: np.ndarray,
              rgba: np.ndarray,
              ax: int, ay: int, aw: int, ah: int,
              feather_px: int) -> np.ndarray:
    H, W = base.shape[:2]
    tw = max(1, min(aw, W - ax))
    th = max(1, min(ah, H - ay))
    if tw <= 0 or th <= 0 or ax < 0 or ay < 0:
        return base

    m  = cv2.resize(rgba, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    x2 = min(W, ax + tw); y2 = min(H, ay + th)
    mw = x2 - ax;         mh = y2 - ay
    if mw <= 0 or mh <= 0:
        return base

    comp  = base.copy()
    rgb_p = m[:mh, :mw, :3].astype(np.float32)
    raw_a = m[:mh, :mw, 3].astype(np.float32)

    # Descarta pixels residuais de pele/borda com alpha fraco.
    raw_a[raw_a < 32] = 0

    alpha_norm = raw_a / 255.0
    alpha  = alpha_norm[:, :, None]
    base_p = comp[ay:y2, ax:x2].astype(np.float32)
    comp[ay:y2, ax:x2] = (rgb_p * alpha + base_p * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    return comp


# ===========================================================================
# Helpers de frame único
# ===========================================================================

def compose_frame(q: int, store: DiskStore, feather_px: int) -> None:
    base                 = store.load_nomouth(q)
    rgba, ax, ay, aw, ah = store.load_mouth_frame(q)
    if rgba.shape[0] <= 4 and rgba.shape[1] <= 4:
        store.save_output(q, base)
        return
    comp = composite(base, rgba, ax, ay, aw, ah, feather_px)
    store.save_output(q, comp)


def _compose_threaded(q: int, store: DiskStore, feather_px: int) -> int:
    try:
        compose_frame(q, store, feather_px)
        return q
    except Exception as exc:
        print(f"[F3-Thread] Erro frame {q}: {exc}")
        try:
            store.save_output(q, store.load_nomouth(q))
        except Exception:
            pass
        return -1


# ===========================================================================
# Fase 3 — composição paralela pura
# ===========================================================================

def run_phase3(store: DiskStore,
               n_frames: int,
               feather_px: int = 0,
               n_workers: int = 4) -> None:
    print(f"\n[F3] Composição paralela — {n_frames} frames, {n_workers} workers...")
    log_step = max(1, n_frames // 10)
    completed = errors = 0

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {
            ex.submit(_compose_threaded, i, store, feather_px): i
            for i in range(n_frames)
        }
        for future in as_completed(futures):
            orig = futures[future]
            try:
                r = future.result()
                if r == -1:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"[F3] EXCEÇÃO frame {orig}: {e}")
                try:
                    store.save_output(orig, store.load_nomouth(orig))
                except Exception:
                    pass
            completed += 1
            if completed % log_step == 0 or completed == n_frames:
                print(f"[F3] {completed}/{n_frames} compostos (erros={errors})")

    cuda_cleanup()
    print(f"[F3] Concluída: {completed} frames, {errors} erros")


# ===========================================================================
# Overlap worker — executa F2+F3 em paralelo com a F1
# ===========================================================================

def _overlap_worker(worker_id: int,
                    ready_queue: Queue,
                    store: DiskStore,
                    mouth_types: List[int],
                    detection_model_path: str,
                    feather_px: int,
                    canonical_cache: Dict,
                    cache_lock: threading.Lock,
                    stop_event: threading.Event,
                    processed_counter: List[int],
                    counter_lock: threading.Lock,
                    upscale_crop_face: float,
                    conf_mouth: float,
                    mouth_padding_per_type: Dict = None,
                    mouth_brightness_per_type: Dict = None,
                    use_rembg: bool = False,
                    use_open_for_half: bool = False,
                    half_open_height_scale: float = 0.55) -> None:
    from ultralytics import YOLO
    try:
        detection_model = YOLO(detection_model_path)
    except Exception as e:
        print(f"[Overlap-{worker_id}] Falha YOLO: {e}")
        return

    while not stop_event.is_set():
        try:
            q = ready_queue.get(timeout=2.0)
        except Empty:
            continue

        if q is _QUEUE_SENTINEL:
            ready_queue.put(_QUEUE_SENTINEL)
            break

        try:
            process_single(q, store, mouth_types, detection_model, canonical_cache,
                           upscale_crop_face, conf_mouth,
                           mouth_padding_per_type, mouth_brightness_per_type,
                           use_rembg=use_rembg,
                           use_open_for_half=use_open_for_half,
                           half_open_height_scale=half_open_height_scale)

            compose_frame(q, store, feather_px)

        except Exception as exc:
            print(f"[Overlap-{worker_id}] Erro frame {q}: {exc}")
            try:
                if not store.has_output(q):
                    store.save_output(q, store.load_nomouth(q))
            except Exception:
                pass

        with counter_lock:
            processed_counter[0] += 1
            if processed_counter[0] % 20 == 0:
                print(f"[Overlap] {processed_counter[0]} frames (worker {worker_id})")

    try:
        del detection_model
    except Exception:
        pass
    cuda_cleanup()
    print(f"[Overlap-{worker_id}] Encerrado.")


def launch_overlap_workers(ready_queue: Queue,
                            store: DiskStore,
                            mouth_types: List[int],
                            detection_model_path: str,
                            feather_px: int,
                            canonical_cache: Dict,
                            vram_safety_mb: int = 1024,
                            upscale_crop_face: float = 2.0,
                            conf_mouth: float = 0.1,
                            mouth_padding_per_type: Dict = None,
                            mouth_brightness_per_type: Dict = None,
                            use_rembg: bool = False,
                            use_open_for_half: bool = False,
                            half_open_height_scale: float = 0.55,
                            ) -> Tuple[List[threading.Thread], threading.Event]:
    import os
    n_workers = max(1, min(
        (os.cpu_count() or 2) // 2,
        _MAX_WORKERS,
        int(max(0., vram_free_mb() - vram_safety_mb) // _VRAM_PER_WORKER_MB),
    ))
    print(f"[Overlap] Lançando {n_workers} workers com YOLO unico "
          f"(upscale=auto(min={upscale_crop_face:.1f}x) conf={conf_mouth} "
          f"padding={mouth_padding_per_type} brightness={mouth_brightness_per_type} "
          f"BEN2={'on' if use_rembg else 'off'} "
          f"half_from_open={'on' if use_open_for_half else 'off'} "
          f"half_height={half_open_height_scale:.2f})...")

    stop_event    = threading.Event()
    cache_lock    = threading.Lock()
    processed_ctr = [0]
    counter_lock  = threading.Lock()
    threads       = []

    for i in range(n_workers):
        t = threading.Thread(
            target=_overlap_worker,
            args=(i, ready_queue, store, mouth_types, detection_model_path,
                  feather_px, canonical_cache, cache_lock,
                  stop_event, processed_ctr, counter_lock,
                  upscale_crop_face, conf_mouth,
                  mouth_padding_per_type, mouth_brightness_per_type,
                  use_rembg, use_open_for_half, half_open_height_scale),
            daemon=True, name=f"overlap-{i}",
        )
        t.start()
        threads.append(t)

    return threads, stop_event


def ensure_all_outputs(store: DiskStore,
                       n_frames: int,
                       feather_px: int = 0) -> None:
    missing = [q for q in range(n_frames) if not store.has_output(q)]
    if not missing:
        return
    print(f"[Fallback] {len(missing)} frames sem output...")
    for q in missing:
        try:
            if store.has_mouth_frame(q):
                compose_frame(q, store, feather_px)
            else:
                store.save_output(q, store.load_nomouth(q))
        except Exception as exc:
            print(f"[Fallback] Frame {q}: {exc}")
            try:
                store.save_output(q, store.load_nomouth(q))
            except Exception:
                pass
