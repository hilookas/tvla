from pathlib import Path
import shutil

from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

def main():
    input_path = "/home/ubuntu/.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned/snapshots/d8419fc249cbb1f29b0c528f05c0d2fe50f46855"
    output_path = "pi05_libero_finetuned_transformers"

    if Path(output_path).exists():
        shutil.rmtree(output_path)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    with safe_open(input_path + "/" + "model.safetensors", framework="pt") as f:
        metadata = f.metadata()

    # Load the safetensors
    state_dict = safe_load_file(input_path + "/" + "model.safetensors")

    # Rename for weight tieing
    # See: https://huggingface.co/docs/safetensors/torch_shared_tensors
    state_dict["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"] = state_dict["model.paligemma_with_expert.paligemma.lm_head.weight"]
    del state_dict["model.paligemma_with_expert.paligemma.lm_head.weight"]
    state_dict["model.paligemma_with_expert.gemma_expert.model.embed_tokens.weight"] = state_dict["model.paligemma_with_expert.gemma_expert.lm_head.weight"]
    del state_dict["model.paligemma_with_expert.gemma_expert.lm_head.weight"]

    # Strip the "model." prefix from all keys in state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_key = k[len("model."):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    # Change the metadata format to "pt"
    metadata["format"] = "pt"

    # Save the new safetensors file in the output_path directory
    out_path = output_path + "/" + "model.safetensors"
    safe_save_file(new_state_dict, out_path, metadata=metadata)

if __name__ == "__main__":
    main()