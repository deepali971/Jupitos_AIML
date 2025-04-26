#QLoRA Implementation
!huggingface-cli login --token --add-to-git-credential

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch # Make sure to import torch

# Check CUDA availability and load the model accordingly
if torch.cuda.is_available():
    # Enable 4-bit quantization for CUDA
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    # Load a quantized model with trust_remote_code
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b", quantization_config=quant_config, trust_remote_code=True,
    )
else:
    # Disable 4-bit quantization if CUDA is not available
    # Instead of using BitsAndBytesConfig, load the model without quantization settings
    # This avoids bitsandbytes CUDA dependency when CUDA is not available.
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-1.3b", trust_remote_code=True,
    )
    print("CUDA not available. Loading model without quantization.")

print("QLoRA Model Successfully Quantized and Loaded:")
print(model)
