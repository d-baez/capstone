import time
from pathlib import Path

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

MODEL_ID = "microsoft/Phi-3.5-vision-instruct"


# macOS: likely CPU or Metal; transformers just cares about "cpu" vs "cuda"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Use float32 on CPU; float16 is usually only safe on GPU
dtype = torch.float16 if device == "cuda" else torch.float32

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
    _attn_implementation="eager",
)

if device == "cpu":
    model.to(device)

# Load image
image_path = "apple.jpeg"  # <-- change this to your file name
image = Image.open(image_path).convert("RGB")

# Prompt in Phi-vision chat format
user_prompt = (
    "Classify the ripeness of the fruit in this image. "
    "Respond with exactly ONE word from this list: "
    "[unripe, ripe, overripe, spoiled]. "
    "Do not add any other words."
)

prompt = f"<|user|>\n<|image_1|>\n{user_prompt}<|end|>\n<|assistant|>\n"

# Prepare inputs
inputs = processor(
    text=prompt,
    images=image,
    return_tensors="pt",
).to(device)

# Generate
print("Starting generation...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=False,
    )
print("Finished generation.")

# Decode only the new tokens
generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Normalize to a single label
label = response.strip().split()[0].lower()
allowed_labels = {"unripe", "ripe", "overripe", "spoiled"}

if label not in allowed_labels:
    label = "unknown"

print("Raw model response:", repr(response))
print("Predicted ripeness label:", label)