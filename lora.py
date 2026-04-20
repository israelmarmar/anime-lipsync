import os
from typing import Any, Tuple

from .utils import get_node_output


def apply_lora(lora_path: str,
               model_obj: Any,
               clip_obj: Any,
               strength_model: float,
               strength_clip: float) -> Tuple[Any, Any]:
    """
    Carrega e aplica uma LoRA via LoraLoader do ComfyUI.
    Aceita caminho completo ou apenas o nome do arquivo.
    String vazia devolve model/clip originais sem modificação.
    """
    if not lora_path or not lora_path.strip():
        return model_obj, clip_obj

    lora_path = lora_path.strip()

    # Resolve pelo nome se o caminho completo não existir
    if not os.path.isfile(lora_path):
        try:
            import folder_paths
            candidates = folder_paths.get_filename_list("loras")
            basename   = os.path.basename(lora_path)
            if basename in candidates:
                lora_path = folder_paths.get_full_path("loras", basename)
            else:
                print(f"[LoRA] Arquivo não encontrado: {lora_path} — ignorando")
                return model_obj, clip_obj
        except Exception:
            print(f"[LoRA] Arquivo não encontrado: {lora_path} — ignorando")
            return model_obj, clip_obj

    try:
        from nodes import NODE_CLASS_MAPPINGS
        loader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        result = loader.load_lora(
            model=model_obj,
            clip=clip_obj,
            lora_name=os.path.basename(lora_path),
            strength_model=strength_model,
            strength_clip=strength_clip,
        )
        model_lora = get_node_output(result, 0)
        clip_lora  = get_node_output(result, 1)
        print(f"[LoRA] Aplicada: {os.path.basename(lora_path)} "
              f"(model={strength_model} clip={strength_clip})")
        return model_lora, clip_lora
    except Exception as exc:
        print(f"[LoRA] ERRO ao aplicar {lora_path}: {exc} — ignorando")
        return model_obj, clip_obj
