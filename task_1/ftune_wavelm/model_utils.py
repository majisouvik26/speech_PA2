from transformers import WavLMModel
from peft import LoraConfig, get_peft_model
import torch

def load_model(device):
    """
    Loads the pre-trained WavLM model.
    """
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    # model.gradient_checkpointing_enable()
    return model

def apply_lora(model):
    """
    Applies LoRA to target modules in the model.
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["attention.q_proj", "attention.k_proj", "attention.v_proj"]
    )
    model = get_peft_model(model, lora_config)
    return model
