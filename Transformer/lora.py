#LoRA Implementation with Output Printing
!huggingface-cli login --token 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 --add-to-git-credential

from peft import LoraConfig, get_peft_model
import torch
import transformers

# Define LoRA configuration
# Change target_modules to ['q_proj', 'v_proj']
config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"
)

# Load a pre-trained model (e.g., LLaMA, Falcon, etc.)
model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")

# Apply LoRA
model = get_peft_model(model, config)

print("LoRA Model Successfully Loaded and Configured:")
print(model)
