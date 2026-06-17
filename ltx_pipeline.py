import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import cv2
import torch
import torch.nn.functional as F

from .store import DiskStore
from .utils import get_node_output, np2t, t2np, cuda_cleanup


def _node(name: str):
    from nodes import NODE_CLASS_MAPPINGS

    cls = NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"Node ComfyUI ausente: {name}")
    return cls()


def _call(obj, method: str, **kwargs):
    fn = getattr(obj, method)
    return fn(**kwargs)


def _exec(obj, **kwargs):
    return _call(obj, "EXECUTE_NORMALIZED", **kwargs)


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, ((int(value) + multiple - 1) // multiple) * multiple)


def _auto_cpu_workers(n_items: int, requested: int = 0, cap: int = 8) -> int:
    if n_items <= 1:
        return 1
    if requested and requested > 0:
        return max(1, min(int(requested), int(n_items)))
    cpu = os.cpu_count() or 2
    return max(1, min(int(n_items), max(1, cpu // 2), int(cap)))


def _resize_image_batch(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if int(images.shape[1]) == int(height) and int(images.shape[2]) == int(width):
        return images
    chw = images.permute(0, 3, 1, 2)
    chw = F.interpolate(chw, size=(int(height), int(width)),
                        mode="bilinear", align_corners=False)
    return chw.permute(0, 2, 3, 1).contiguous()


def _face_crop_from_nomouth(store: DiskStore, idx: int):
    nomouth = store.load_nomouth(idx)
    bbox = store.load_face_bbox(idx)
    if bbox is None:
        return nomouth
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = nomouth.shape[:2]
    x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return nomouth
    return nomouth[y1:y2, x1:x2]


def _face_crop_batch_from_nomouth(store: DiskStore, n_frames: int,
                                  n_workers: int = 0) -> torch.Tensor:
    workers = _auto_cpu_workers(n_frames, n_workers)
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            crops = list(ex.map(lambda i: _face_crop_from_nomouth(store, i), range(n_frames)))
    else:
        crops = [_face_crop_from_nomouth(store, i) for i in range(n_frames)]
    max_h = max(max(1, c.shape[0]) for c in crops)
    max_w = max(max(1, c.shape[1]) for c in crops)
    target_w = _ceil_to_multiple(max_w, 64)
    target_h = _ceil_to_multiple(max_h, 64)
    print(f"[LTX] Face crops → batch {target_w}x{target_h} "
          f"(max crop {max_w}x{max_h})")
    frames = []
    def _resize_to_tensor(crop):
        if crop.shape[1] != target_w or crop.shape[0] != target_h:
            crop = cv2.resize(crop, (target_w, target_h),
                              interpolation=cv2.INTER_LANCZOS4)
        return np2t(crop)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            frames = list(ex.map(_resize_to_tensor, crops))
    else:
        frames = [_resize_to_tensor(crop) for crop in crops]
    return torch.cat(frames, dim=0)


def run_ltx_lipsync(
    store: DiskStore,
    n_frames: int,
    fps: int,
    audio,
    prompt: str,
    negative_prompt: str,
    z_prompt: str,
    seed: int = 0,
    ltx_steps: int = 8,
    ltx_cfg: float = 1.2,
    ltx_iclora_strength: float = 0.9,
    ltx_image_strength: float = 0.7,
    ltx_canny_strength: float = 0.5,
    z_steps: int = 9,
    z_cfg: float = 1.0,
    z_denoise: float = 0.9,
    z_controlnet_strength: float = 0.7,
    ltx_sigmas: str = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
    ltx_clip_name1: str = "gemma_3_12B_it_fp4_mixed.safetensors",
    ltx_clip_name2: str = "ltx-2.3_text_projection_bf16.safetensors",
    ltx_unet_name: str = "ltx-2.3-22b-distilled-fp8.safetensors",
    ltx_video_vae_name: str = "LTX23_video_vae_bf16.safetensors",
    ltx_audio_vae_name: str = "LTX23_audio_vae_bf16.safetensors",
    ltx_iclora_name: str = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    z_unet_name: str = "z_image_turbo_bf16.safetensors",
    z_clip_name: str = "qwen_3_4b.safetensors",
    z_vae_name: str = "ae.safetensors",
    z_patch_name: str = "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors",
    cpu_workers: int = 0,
) -> torch.Tensor:
    """
    Gera a animação inteira com LTX. Z-Image Turbo é usado apenas para refinar
    o primeiro frame sem boca, que vira a referência visual principal do LTX.
    """
    if n_frames <= 0:
        raise ValueError("n_frames deve ser > 0")

    seed = int(seed) if int(seed) > 0 else random.randint(1, 2**63 - 1)
    print(f"[LTX] Preparando batch de crops da face sem boca ({n_frames} frames)...")
    face_crop_images = _face_crop_batch_from_nomouth(store, n_frames, n_workers=cpu_workers)
    first_frame = face_crop_images[:1]

    randomnoise = _node("RandomNoise")
    ksamplerselect = _node("KSamplerSelect")
    manualsigmas = _node("ManualSigmas")
    dualcliploader = _node("DualCLIPLoader")
    cliptextencode = _node("CLIPTextEncode")
    unetloader = _node("UNETLoader")
    vaeloaderkj = _node("VAELoaderKJ")
    vaeloader = _node("VAELoader")
    cliploader = _node("CLIPLoader")
    modelpatchloader = _node("ModelPatchLoader")
    cannyedgepreprocessor = _node("CannyEdgePreprocessor")
    qwenimagediffsynthcontrolnet = _node("QwenImageDiffsynthControlnet")
    modelsamplingauraflow = _node("ModelSamplingAuraFlow")
    conditioningzeroout = _node("ConditioningZeroOut")
    ksampler = _node("KSampler")
    vaedecode = _node("VAEDecode")
    ltxvpreprocess = _node("LTXVPreprocess")
    emptyltxvlatentvideo = _node("EmptyLTXVLatentVideo")
    ltxvimgtovideoconditiononly = _node("LTXVImgToVideoConditionOnly")
    ltxaddvideoicloraguide = _node("LTXAddVideoICLoRAGuide")
    cfgguider = _node("CFGGuider")
    jwfloattointeger = _node("JWFloatToInteger")
    ltxvaudiovaeencode = _node("LTXVAudioVAEEncode")
    ltxsetaudioreftokens = _node("LTXVSetAudioRefTokens")
    ltxvconcatavlatent = _node("LTXVConcatAVLatent")
    samplercustomadvanced = _node("SamplerCustomAdvanced")
    ltxvseparateavlatent = _node("LTXVSeparateAVLatent")
    ltxvcropguides = _node("LTXVCropGuides")
    easy_cleangpuused = _node("easy cleanGpuUsed")

    print("[LTX] Carregando modelos LTX/Z...")
    ltx_clip = dualcliploader.load_clip(
        clip_name1=ltx_clip_name1,
        clip_name2=ltx_clip_name2,
        type="ltxv",
        device="default",
    )
    pos = cliptextencode.encode(text=prompt, clip=get_node_output(ltx_clip, 0))
    neg = cliptextencode.encode(text=negative_prompt, clip=get_node_output(ltx_clip, 0))
    ltx_model = unetloader.load_unet(unet_name=ltx_unet_name, weight_dtype="default")
    ltx_vae = vaeloaderkj.load_vae(
        vae_name=ltx_video_vae_name,
        device="main_device",
        weight_dtype="bf16",
    )
    audio_vae = vaeloader.load_vae(vae_name=ltx_audio_vae_name)

    z_model = unetloader.load_unet(unet_name=z_unet_name, weight_dtype="default")
    z_clip = cliploader.load_clip(clip_name=z_clip_name, type="lumina2", device="default")
    z_vae = vaeloader.load_vae(vae_name=z_vae_name)
    z_patch = modelpatchloader.load_model_patch(name=z_patch_name)
    z_pos = cliptextencode.encode(text=z_prompt, clip=get_node_output(z_clip, 0))

    print("[LTX] Refinando primeiro frame com Z-Image Turbo...")
    z_canny = cannyedgepreprocessor.execute(
        low_threshold=100,
        high_threshold=200,
        resolution=512,
        image=first_frame,
    )
    z_control = qwenimagediffsynthcontrolnet.diffsynth_controlnet(
        strength=float(z_controlnet_strength),
        model=get_node_output(z_model, 0),
        model_patch=get_node_output(z_patch, 0),
        vae=get_node_output(z_vae, 0),
        image=get_node_output(z_canny, 0),
    )
    z_model_patched = modelsamplingauraflow.patch_aura(
        shift=3,
        model=get_node_output(z_control, 0),
    )
    z_neg = conditioningzeroout.zero_out(conditioning=get_node_output(z_pos, 0))
    z_latent = _node("VAEEncode").encode(pixels=first_frame, vae=get_node_output(z_vae, 0))
    z_sample = ksampler.sample(
        seed=seed + 17,
        steps=int(z_steps),
        cfg=float(z_cfg),
        sampler_name="res_multistep",
        scheduler="simple",
        denoise=float(z_denoise),
        model=get_node_output(z_model_patched, 0),
        positive=get_node_output(z_pos, 0),
        negative=get_node_output(z_neg, 0),
        latent_image=get_node_output(z_latent, 0),
    )
    z_ref = vaedecode.decode(samples=get_node_output(z_sample, 0), vae=get_node_output(z_vae, 0))
    ltx_ref = ltxvpreprocess.EXECUTE_NORMALIZED(
        img_compression=0,
        image=get_node_output(z_ref, 0),
    )

    print("[LTX] Preparando guias Canny no_mouth e latentes AV...")
    canny_sequence = cannyedgepreprocessor.execute(
        low_threshold=100,
        high_threshold=200,
        resolution=512,
        image=face_crop_images,
    )
    canny_images = get_node_output(canny_sequence, 0)
    length = int(canny_images.shape[0])
    height = int(canny_images.shape[1])
    width = int(canny_images.shape[2])

    latent_video = emptyltxvlatentvideo.EXECUTE_NORMALIZED(
        width=width,
        height=height,
        length=length,
        batch_size=1,
    )
    image_condition = ltxvimgtovideoconditiononly.generate(
        strength=float(ltx_image_strength),
        bypass=False,
        vae=get_node_output(ltx_vae, 0),
        image=get_node_output(ltx_ref, 0),
        latent=get_node_output(latent_video, 0),
    )
    ltx_lora = _exec(
        _node("LTXICLoRALoaderModelOnly"),
        lora_name=ltx_iclora_name,
        strength_model=float(ltx_iclora_strength),
        model=get_node_output(ltx_model, 0),
    )
    ltx_cond = _exec(
        _node("LTXVConditioning"),
        frame_rate=float(fps),
        positive=get_node_output(pos, 0),
        negative=get_node_output(neg, 0),
    )
    encoded_audio = ltxvaudiovaeencode.EXECUTE_NORMALIZED(
        audio=audio,
        audio_vae=get_node_output(audio_vae, 0),
    )
    audio_condition = ltxsetaudioreftokens.EXECUTE_NORMALIZED(
        positive=get_node_output(ltx_cond, 0),
        negative=get_node_output(ltx_cond, 1),
        audio_latent=get_node_output(encoded_audio, 0),
    )
    guide = ltxaddvideoicloraguide.EXECUTE_NORMALIZED(
        frame_idx=0,
        strength=float(ltx_canny_strength),
        latent_downscale_factor=get_node_output(ltx_lora, 1),
        crop="disabled",
        use_tiled_encode=False,
        tile_size=256,
        tile_overlap=64,
        positive=get_node_output(audio_condition, 0),
        negative=get_node_output(audio_condition, 1),
        vae=get_node_output(ltx_vae, 0),
        latent=get_node_output(image_condition, 0),
        image=canny_images,
    )
    guider = cfgguider.EXECUTE_NORMALIZED(
        cfg=float(ltx_cfg),
        model=get_node_output(ltx_lora, 0),
        positive=get_node_output(guide, 0),
        negative=get_node_output(guide, 1),
    )
    audio_latent = get_node_output(audio_condition, 2)
    if audio_latent is None:
        audio_latent = get_node_output(encoded_audio, 0)
    av_latent = ltxvconcatavlatent.EXECUTE_NORMALIZED(
        video_latent=get_node_output(guide, 2),
        audio_latent=audio_latent,
    )
    noise = randomnoise.EXECUTE_NORMALIZED(noise_seed=seed)
    sampler = ksamplerselect.EXECUTE_NORMALIZED(sampler_name="euler_ancestral_cfg_pp")
    sigmas = manualsigmas.EXECUTE_NORMALIZED(sigmas=ltx_sigmas)

    print("[LTX] Amostrando animação LTX...")
    sampled = samplercustomadvanced.EXECUTE_NORMALIZED(
        noise=get_node_output(noise, 0),
        guider=get_node_output(guider, 0),
        sampler=get_node_output(sampler, 0),
        sigmas=get_node_output(sigmas, 0),
        latent_image=get_node_output(av_latent, 0),
    )
    separated = ltxvseparateavlatent.EXECUTE_NORMALIZED(
        av_latent=get_node_output(sampled, 0)
    )
    cropped = ltxvcropguides.EXECUTE_NORMALIZED(
        positive=get_node_output(guide, 0),
        negative=get_node_output(guide, 1),
        latent=get_node_output(separated, 0),
    )
    cleaned = easy_cleangpuused.EXECUTE_NORMALIZED(
        anything=get_node_output(cropped, 2),
        unique_id=random.randint(1, 2**63 - 1),
    )
    decoded = vaedecode.decode(
        samples=get_node_output(cleaned, 0),
        vae=get_node_output(ltx_vae, 0),
    )
    images = get_node_output(decoded, 0)
    cuda_cleanup()
    return images


def save_ltx_output_frames(store: DiskStore, images: torch.Tensor, n_frames: Optional[int] = None) -> int:
    count = int(images.shape[0])
    if n_frames is not None:
        count = min(count, int(n_frames))
    for i in range(count):
        store.save_output(i, t2np(images[i:i + 1]))
    return count


def save_ltx_face_frames(store: DiskStore,
                         images: torch.Tensor,
                         mouth_types,
                         n_frames: Optional[int] = None,
                         n_workers: int = 0) -> int:
    count = int(images.shape[0])
    if n_frames is not None:
        count = min(count, int(n_frames))

    images_np = (images[:count].detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    workers = _auto_cpu_workers(count, n_workers)

    def _save_one(i: int) -> None:
        face = images_np[i]
        bbox = store.load_face_bbox(i)
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop_w = max(1, x2 - x1)
            crop_h = max(1, y2 - y1)
        else:
            ref = store.load_nomouth(i)
            crop_h, crop_w = ref.shape[:2]
        if face.shape[1] != crop_w or face.shape[0] != crop_h:
            face = cv2.resize(face, (crop_w, crop_h),
                              interpolation=cv2.INTER_LANCZOS4)
        mt = int(mouth_types[i]) if i < len(mouth_types) else 0
        store.save_face_frame(
            i, face, mt, 0, "ltx_face_animation", "ltx",
            crop_w=crop_w, crop_h=crop_h, mouth_cache_epoch=0,
            use_mouth_cache=False,
        )

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_save_one, i) for i in range(count)]
            for future in as_completed(futures):
                future.result()
    else:
        for i in range(count):
            _save_one(i)
    return count
