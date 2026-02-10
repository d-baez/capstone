import sys
import time
from pathlib import Path

from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import torch

MODEL_ID = "microsoft/Phi-3.5-vision-instruct"

print(torch.backends.mps.is_available())

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
    _attn_implementation="eager",
)
model.to(device)
model.eval()

if device == "cpu":
    model.to(device)

# # Load image
# image_path = "apple.jpeg"  # <-- change this to your file name
# image = Image.open(image_path).convert("RGB")

# Prompt in Phi-vision chat format
user_prompt = (
    "Classify the ripeness of the fruit in this image. "
    "Respond with exactly ONE word and accuracy percentage from this list: "
    "[unripe, ripe, overripe, spoiled]. "
    "Do not add any other words."
    #  """
    # You are a fruit ripeness inspector.
    #
    # Look at the image and:
    # 1. Identify the type of fruit.
    # 2. Say whether it is unripe, ripe, or overripe.
    # 3. Briefly explain your reasoning.
    #
    # Format:
    # fruit: <type>
    # ripeness: <unripe|ripe|overripe|spoiled>
    # """
)
allowed_labels = {"unripe", "ripe", "overripe", "spoiled"}

def classify_image(image_path: str):
    """
        Run the model on a single image and return (raw_response, label).
        Uses global `model`, `processor`, `device`, `user_prompt`.
        """
    image = Image.open(image_path).convert("RGB")

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
            max_new_tokens=8,
            do_sample=False,
        )
    print("Finished generation.")

    # Decode only the new tokens
    generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Normalize to a single label
    label = response.strip().split()[0].lower()
    if label not in allowed_labels:
        label = "unknown"
    return response, label

def timed_single_inference(image_path: str):
    start = time.time()
    raw, label = classify_image(image_path)
    elapsed = time.time() - start

    print(f"\nImage: {image_path}")
    print("Raw model response:", repr(raw))
    print("Predicted ripeness label:", label)
    print(f"Inference time: {elapsed:.3f} seconds")

    return raw, label, elapsed

def batch_test_folder(folder_path: str):
    """
    Run inference on all images in a folder and print summary stats.
    """
    folder = Path(folder_path)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    image_paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in exts
    )

    if not image_paths:
        print(f"No images found in {folder}")
        return

    print(f"\nFound {len(image_paths)} images in {folder}")
    times = []
    results = []

    for img_path in image_paths:
        print("-" * 60)
        try:
            raw, label, elapsed = timed_single_inference(str(img_path))
            times.append(elapsed)
            results.append((img_path.name, label, elapsed))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if times:
        avg_time = sum(times) / len(times)
        print("\n=== Batch Summary ===")
        print(f"Images processed: {len(times)}")
        print(f"Average inference time: {avg_time:.3f} seconds")
        print(f"Min time: {min(times):.3f} seconds")
        print(f"Max time: {max(times):.3f} seconds")

        print("\nPer-image results:")
        for name, label, t in results:
            print(f"{name:30s} -> {label:8s} ({t:.3f}s)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phi-3.5 Vision ripeness classifier (single image or folder)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="apple.jpeg",  # fallback so it still works with no args
        help="Path to an image file or a folder containing images",
    )
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        batch_test_folder(str(p))
    else:
        timed_single_inference(str(p))
