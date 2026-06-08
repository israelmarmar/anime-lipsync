# Anime Lip Sync Custom Node

Custom node para ComfyUI focado em lip sync de vídeos/animações estilo anime.
O node recebe áudio e uma sequência de frames, detecta fonemas, gera variações de boca com diffusion e recompõe a boca no vídeo final.

O node principal aparece como:

```text
LipSync / LipSync Pipeline V6
```

Ele retorna:

- `video_path`: caminho do MP4 final.
- `video_info`: metadados do vídeo para integrações como VHS.

## Visão Geral

O pipeline trabalha em quatro fases:

1. **Fase 0 - detecção e preparação**
   - Salva os frames de entrada em disco.
   - Usa um único modelo YOLO para detectar:
     - classe `0`: face
     - classe `1`: boca
   - Remove/inpinta a boca original quando `remove_mouth=True`.
   - Salva bbox da face, máscara da boca e frames sem boca.

2. **Fase 1 - geração das faces/bocas**
   - Converte o áudio em fonemas usando Allosaurus.
   - Mapeia fonemas para três tipos de boca:
     - `neutral_closed`
     - `half_open`
     - `fully_open`
   - Agrupa faces por similaridade usando um mapa de **scribble**/contornos da face, mais preciso para variações de pose e linhas de anime.
   - Usa `controlnet_aux.HEDdetector` em modo **scribble** para gerar o mapa de controle, com fallback para HEDPreprocessor, e então ControlNet + Z-Image Turbo/Qwen DiffSynth para gerar as faces com a boca correta.

3. **Fase 2 - recorte da boca gerada**
   - Detecta a boca gerada frame a frame com o mesmo YOLO.
   - Refina o alpha da boca:
     - boca fechada preserva principalmente linhas dos lábios, evitando manchas de pele;
     - bocas abertas usam máscara por cor, borda, elipse e `rembg` quando disponível.
   - Suaviza temporalmente a posição/tamanho da boca em coordenadas relativas ao rosto, reduzindo tremedeira.

4. **Fase 3 - composição**
   - Recompõe o recorte RGBA da boca sobre o frame sem boca.
   - Remove pixels de alpha fraco para reduzir halos e preenchimento de pele.
   - Gera o MP4 final com o áudio original.

## Modelo YOLO Único

Este node espera um único modelo YOLO treinado com duas classes:

```text
0 = face
1 = mouth
```

Campo:

```text
detection_model_path
```

O default tenta usar:

```text
best.pt
```

Se esse arquivo não existir, usa:

```text
best.pt
```

dentro da pasta do custom node.

## Workflow de Exemplo

Há um workflow de exemplo em:

```text
examples/anime_lip_sync_v7.json
```

Ele usa:

- frames/imagens de entrada;
- áudio;
- `UNETLoader` para `z_image_turbo_bf16.safetensors`;
- `CLIPLoader`;
- `VAELoader`;
- prompts externos opcionais para cada tipo de boca;
- o node `LipSync Pipeline V6`.

## Entradas Principais

### Obrigatórias

- `audio`: áudio de entrada.
- `images`: sequência de frames do vídeo.
- `model`: modelo diffusion.
- `model_patch`: patch/controlnet model.
- `clip`: CLIP usado nos prompts.
- `vae`: VAE.
- `detection_model_path`: YOLO único face+boca.

### Gerais

- `fps`: FPS do vídeo final e da análise de fonemas.
- `source_fps`: FPS da sequência original de imagens.
- `lang_id`: idioma para Allosaurus. O default `uni` costuma ser o mais flexível.
- `sim_threshold`: threshold para separar grupos de face. Agora usa comparação por scribble.
- `vram_safety_margin_mb`: margem de VRAM antes de lançar workers.
- `enable_overlap`: processa F2/F3 em paralelo com a F1 quando possível.
- `video_output_filename`: nome base do MP4 final.

### Detecção e Máscara

- `mouth_conf`: confiança mínima para detecção da boca no YOLO.
- `upscale_crop_face`: upscale mínimo do crop da face antes de detectar a boca.
- `remove_mouth`: remove a boca original antes da diffusion.
- `hed_detector_mode`: fonte do mapa HED/scribble para o ControlNet:
  - `auto`: tenta `controlnet_aux.HEDdetector(scribble=True)` e cai para `HEDPreprocessor`.
  - `controlnet_aux`: força `controlnet_aux.HEDdetector`; falha se a dependência/modelo não estiver disponível.
  - `comfy_hed`: força o `HEDPreprocessor` do ComfyUI.
- `mask_dilation`: expansão da máscara de inpaint da boca original.
- `mask_blur`: suavização da máscara no inpaint.
- `compose_feather_px`: feather da composição final.

### Ajustes por Tipo de Boca

- `mouth_padding_closed`
- `mouth_padding_half`
- `mouth_padding_open`

Use padding baixo para evitar carregar pele ao redor da boca. O pipeline já aplica padding automático pequeno por tipo.

- `mouth_brightness_closed`
- `mouth_brightness_half`
- `mouth_brightness_open`

Use para corrigir boca gerada escura/clara demais antes da composição.

### Florence-2 para Personagem Específico

Quando há mais de um personagem no frame:

- `enable_character_detect=True`
- `character_query`: descrição do personagem, por exemplo `the girl with blue hair`
- `character_margin`: margem ao redor do bbox detectado
- `florence_model_id`: default `microsoft/Florence-2-base`

O Florence-2 localiza o personagem antes do YOLO, restringindo a busca da face.

## LoRAs e KSampler

Cada tipo de boca pode usar LoRA e parâmetros de sampling próprios:

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

Também é possível fornecer `CONDITIONING` externo:

- `cond_closed`
- `cond_half`
- `cond_open`

Quando um conditioning externo é conectado, ele substitui o prompt interno daquele tipo de boca.

## Dicas de Ajuste

### A boca está tremendo

- O pipeline já aplica suavização temporal relativa à face.
- Se ainda tremer:
  - aumente um pouco `mouth_conf` para evitar detecções instáveis;
  - reduza `upscale_crop_face` se o detector estiver pegando detalhes demais;
  - reduza mudanças bruscas nos prompts/LoRAs entre tipos de boca.

### A boca aberta está com pele ao redor

- Reduza `mouth_padding_half` e `mouth_padding_open`.
- Aumente levemente `compose_feather_px` apenas se a borda estiver dura.
- Evite brilho muito alto em `mouth_brightness_half/open`, porque ele pode clarear pele residual.

### A boca fechada aparece como uma mancha de pele

- Use `mouth_padding_closed=0` ou um valor baixo.
- A máscara de boca fechada preserva principalmente linhas; se a linha sumir, aumente pouco a força/qualidade da LoRA de boca fechada em vez de aumentar padding.

### O personagem muda ou troca de rosto

- Ajuste `sim_threshold`.
- Valores mais altos criam mais grupos de face.
- Valores mais baixos reutilizam mais faces geradas.
- A comparação usa scribble/contornos, então mudanças de pose e linhas faciais contam mais que cor.

### Mais de um personagem no vídeo

Ative Florence-2 e preencha `character_query`. Sem isso, o YOLO pode escolher a face com maior confiança no frame.

## Dependências

Além do ComfyUI e dos nodes usados pelo workflow, o pipeline usa:

- `ultralytics` para YOLO.
- `opencv-python` / `cv2`.
- `numpy`.
- `torch`.
- `Pillow`.
- `allosaurus` para fonemas.
- `controlnet_aux` opcional/recomendado para HEDdetector em modo scribble.
- `skimage` opcional para SSIM.
- `rembg` opcional para refino de alpha das bocas abertas.
- `transformers`, `timm`, `einops` se usar Florence-2.

O node também espera que os nodes ComfyUI abaixo existam:

- `HEDPreprocessor`
- `QwenImageDiffsynthControlnet`
- `VAEEncode`
- `VAEDecode`
- `KSampler`
- `CLIPTextEncode`

## Estrutura dos Arquivos

- `node.py`: entrada do custom node e orquestração geral.
- `phase0.py`: YOLO face+boca, inpaint da boca original e bbox da face.
- `phase1.py`: fonemas, agrupamento por scribble, HED/ControlNet/diffusion.
- `phase2.py`: detecção/recorte da boca gerada, alpha e suavização temporal.
- `phase3.py`: composição final e workers de overlap.
- `phonemes.py`: integração Allosaurus.
- `character_detect.py`: seleção de personagem com Florence-2.
- `store.py`: armazenamento temporário em disco.
- `constants.py`: prompts, tipos de boca e mapeamento fonema -> boca.
- `examples/anime_lip_sync_v7.json`: workflow de exemplo.

## Saída

O vídeo final é salvo no diretório de output do ComfyUI com o nome definido por:

```text
video_output_filename
```

O node retorna o caminho do MP4 e as informações de vídeo para a UI.
