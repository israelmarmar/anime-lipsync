# Anime Lip Sync Custom Node

Custom node for ComfyUI focused on lip sync for anime-style videos/animations.
The node receives audio and a sequence of frames, detects phonemes, generates mouth variations with diffusion, and recomposes the mouth into the final video.

The main node appears as:

```text
LipSync / LipSync Pipeline V6
```

It returns:

- `video_path`: path to the final MP4.
- `video_info`: video metadata for integrations such as VHS.
- `debug_dir`: path to the persisted debug folder when `save_debug_folder=True`.

## Overview

The pipeline works in four phases:

1. **Phase 0 - detection and preparation**
   - Saves the input frames to disk.
   - Uses a single YOLO model to detect:
     - class `0`: face
     - class `1`: mouth
   - Removes/inpaints the original mouth when `remove_mouth=True`.
   - Saves the face bbox, mouth mask, and mouthless frames.

2. **Phase 1 - face/mouth generation**
   - Converts audio into phonemes using Allosaurus.
   - Maps phonemes to three mouth types:
     - `neutral_closed`
     - `half_open`
     - `fully_open`
   - Groups faces by similarity using a face **scribble**/contour map, which is more accurate for pose variations and anime line art.
   - Uses `controlnet_aux.HEDdetector` in **scribble** mode to generate the control map, with a fallback to HEDPreprocessor, then uses ControlNet + Z-Image Turbo/Qwen DiffSynth to generate faces with the correct mouth.

3. **Phase 2 - generated mouth crop**
   - Detects the generated mouth frame by frame with the same YOLO model.
   - Refines the mouth alpha:
     - closed mouths mainly preserve lip lines, avoiding skin smudges;
     - open mouths use masks based on color, edge, ellipse, and `rembg` when available.
   - Temporally smooths the mouth position/size in coordinates relative to the face, reducing jitter.

4. **Phase 3 - composition**
   - Recomposes the RGBA mouth crop over the mouthless frame.
   - Removes weak-alpha pixels to reduce halos and skin fill.
   - Generates the final MP4 with the original audio.

## Single YOLO Model

This node expects a single YOLO model trained with two classes:

```text
0 = face
1 = mouth
```

Field:

```text
detection_model_path
```

The default tries to use:

[https://huggingface.co/israelmarmar/anime_face_mouth/blob/main/anime_face_mouth.pt](https://huggingface.co/israelmarmar/anime_face_mouth/blob/main/anime_face_mouth.pt)

inside the custom node folder.

## Example Workflow

There is an example workflow at:

```text
examples/anime_lip_sync_v14.json
```

It uses:

- input frames/images;
- audio;
- `UNETLoader` for `z_image_turbo_bf16.safetensors`;
- `CLIPLoader`;
- `VAELoader`;
- optional external prompts for each mouth type;
- the `LipSync Pipeline V6` node.

## Main Inputs

### Required

- `audio`: input audio.
- `images`: video frame sequence.
- `model`: diffusion model.
- `model_patch`: patch/controlnet model.
- `clip`: CLIP used in prompts.
- `vae`: VAE.
- `detection_model_path`: single face+mouth YOLO model.

### General

- `fps`: FPS for the final video and phoneme analysis.
- `source_fps`: FPS of the original image sequence.
- `lang_id`: language for Allosaurus. The default `uni` is usually the most flexible.
- `sim_threshold`: threshold for separating face groups. It now uses scribble comparison.
- `vram_safety_margin_mb`: VRAM margin before launching workers.
- `enable_overlap`: processes F2/F3 in parallel with F1 when possible.
- `save_debug_folder`: copies the temporary work folder to the ComfyUI output and returns its path in `debug_dir`.
- `verify_generated_mouth`: uses YOLO to validate whether the generated face contains a mouth between the nose and chin; for `neutral_closed`, it first removes skin with BEN2 and validates only the vertical thickness of the resulting alpha.
- `mouth_regen_attempts`: number of KSampler rerenders when the mouth is not detected in the generated face.
- `use_open_for_half`: uses the `fully_open` generation on `half_open` frames and vertically shrinks only the mouth RGBA crop.
- `half_open_height_scale`: scale of the open mouth height used as `half_open`; `0.55` keeps 55% of the original height.
- `video_output_filename`: base name for the final MP4.

### Detection and Mask

- `mouth_conf`: minimum confidence for YOLO mouth detection.
- `upscale_crop_face`: minimum upscale for the face crop before detecting the mouth.
- `remove_mouth`: removes the original mouth before diffusion.
- `hed_detector_mode`: source of the HED/scribble map for ControlNet:
  - `auto`: tries `controlnet_aux.HEDdetector(scribble=True)` and falls back to `HEDPreprocessor`.
  - `controlnet_aux`: forces `controlnet_aux.HEDdetector`; fails if the dependency/model is not available.
  - `comfy_hed`: forces ComfyUI's `HEDPreprocessor`.
- `mask_dilation`: expansion of the original mouth inpaint mask.
- `mask_blur`: mask smoothing for inpaint.
- `compose_feather_px`: feathering for the final composition.

### Per-Mouth-Type Adjustments

- `mouth_padding_closed`
- `mouth_padding_half`
- `mouth_padding_open`

Use low padding to avoid carrying skin around the mouth. The pipeline already applies a small automatic padding per type.

- `mouth_brightness_closed`
- `mouth_brightness_half`
- `mouth_brightness_open`

Use these to correct a generated mouth that is too dark or too bright before composition.

### Florence-2 for a Specific Character

When there is more than one character in the frame:

- `enable_character_detect=True`
- `character_query`: character description, for example `the girl with blue hair`
- `character_margin`: margin around the detected bbox
- `florence_model_id`: default `microsoft/Florence-2-base`

Florence-2 locates the character before YOLO, restricting the face search.

## LoRAs and KSampler

Each mouth type can use its own LoRA and sampling parameters:

- closed:
  - `lora_closed_path`
  - `closed_steps`
  - `closed_denoise`
  - `closed_cfg`
  - `closed_cn_strength`

- half:
  - `lora_half_path`
  - `half_steps`
  - `half_denoise`
  - `half_cfg`
  - `half_cn_strength`

- open:
  - `lora_open_path`
  - `open_steps`
  - `open_denoise`
  - `open_cfg`
  - `open_cn_strength`

You can also provide external `CONDITIONING`:

- `cond_closed`
- `cond_half`
- `cond_open`

When external conditioning is connected, it replaces the internal prompt for that mouth type.

## Tuning Tips

### The mouth is jittering

- The pipeline already applies temporal smoothing relative to the face.
- If it still jitters:
  - slightly increase `mouth_conf` to avoid unstable detections;
  - reduce `upscale_crop_face` if the detector is picking up too many details;
  - reduce abrupt changes in prompts/LoRAs between mouth types.

### The open mouth has skin around it

- Reduce `mouth_padding_half` and `mouth_padding_open`.
- Slightly increase `compose_feather_px` only if the edge is too harsh.
- Avoid very high brightness in `mouth_brightness_half/open`, because it can brighten residual skin.

### The closed mouth appears as a skin smudge

- Use `mouth_padding_closed=0` or a low value.
- The closed-mouth mask mainly preserves lines; if the line disappears, slightly increase the closed-mouth LoRA strength/quality instead of increasing padding.

### The character changes or the face swaps

- Adjust `sim_threshold`.
- Higher values create more face groups.
- Lower values reuse more generated faces.
- The comparison uses scribble/contours, so pose changes and facial lines matter more than color.

### More than one character in the video

Enable Florence-2 and fill in `character_query`. Without this, YOLO may choose the face with the highest confidence in the frame.

## Dependencies

In addition to ComfyUI and the nodes used by the workflow, the pipeline uses:

- `ultralytics` for YOLO.
- `opencv-python` / `cv2`.
- `numpy`.
- `torch`.
- `Pillow`.
- `allosaurus` for phonemes.
- `controlnet_aux` optional/recommended for HEDdetector in scribble mode.
- `skimage` optional for SSIM.
- `rembg` optional for open-mouth alpha refinement.
- `transformers`, `timm`, `einops` if using Florence-2.

The node also expects the following ComfyUI nodes to exist:

- `HEDPreprocessor`
- `QwenImageDiffsynthControlnet`
- `VAEEncode`
- `VAEDecode`
- `KSampler`
- `CLIPTextEncode`

## File Structure

- `node.py`: custom node entry point and general orchestration.
- `phase0.py`: YOLO face+mouth, original mouth inpaint, and face bbox.
- `phase1.py`: phonemes, scribble-based grouping, HED/ControlNet/diffusion.
- `phase2.py`: generated mouth detection/crop, alpha, and temporal smoothing.
- `phase3.py`: final composition and overlap workers.
- `phonemes.py`: Allosaurus integration.
- `character_detect.py`: character selection with Florence-2.
- `store.py`: temporary on-disk storage.
- `constants.py`: prompts, mouth types, and phoneme -> mouth mapping.
- `examples/anime_lip_sync_v7.json`: example workflow.

## Output

The final video is saved in the ComfyUI output directory with the name defined by:

```text
video_output_filename
```

The node returns the MP4 path and video information for the UI.
