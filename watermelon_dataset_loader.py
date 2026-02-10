# watermelon_dataset_loader.py
import os
import shutil
from pathlib import Path


def organize_watermelon_dataset(
        qilin_dataset_dir,
        output_dir="data_watermelon",
        sweetness_thresholds=(9.0, 11.0, 13.0)
):
    """
    Organize Qilin watermelon dataset into train/val structure
    compatible with your banana training pipeline.

    Args:
        qilin_dataset_dir: Path to the "19_datasets" folder
        output_dir: Where to create organized dataset
        sweetness_thresholds: (unripe_max, ripe_max, overripe_max)
    """
    output_path = Path(output_dir)
    train_path = output_path / "train"
    val_path = output_path / "val"

    # Create class directories
    class_names = ["unripe", "ripe", "overripe", "spoiled"]
    for split in [train_path, val_path]:
        for class_name in class_names:
            (split / class_name).mkdir(parents=True, exist_ok=True)

    # Process each watermelon sample
    subdirs = [d for d in os.listdir(qilin_dataset_dir)
               if os.path.isdir(os.path.join(qilin_dataset_dir, d))]

    all_samples = []

    for subdir in subdirs:
        data_id, sweetness_str = subdir.split("_")
        sweetness = float(sweetness_str)

        # Classify based on sweetness
        if sweetness < sweetness_thresholds[0]:
            class_label = "unripe"
        elif sweetness < sweetness_thresholds[1]:
            class_label = "ripe"
        elif sweetness < sweetness_thresholds[2]:
            class_label = "overripe"
        else:
            class_label = "spoiled"

        # Find all images in chu folder
        chu_path = os.path.join(qilin_dataset_dir, subdir, "chu")
        if not os.path.exists(chu_path):
            continue

        folders = [f for f in os.listdir(chu_path)
                   if os.path.isdir(os.path.join(chu_path, f))]

        for folder in folders:
            folder_path = os.path.join(chu_path, folder)
            jpg_files = [f for f in os.listdir(folder_path)
                         if f.endswith(".jpg")]

            for jpg_file in jpg_files:
                jpg_path = os.path.join(folder_path, jpg_file)
                all_samples.append((jpg_path, class_label, data_id, folder))

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(all_samples)

    split_idx = int(0.8 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    # Copy files
    print(f"Organizing {len(train_samples)} training and {len(val_samples)} validation images...")

    for samples, split_path in [(train_samples, train_path), (val_samples, val_path)]:
        for idx, (jpg_path, class_label, data_id, folder) in enumerate(samples):
            dest_filename = f"{data_id}_{folder}_{idx}.jpg"
            dest_path = split_path / class_label / dest_filename
            shutil.copy2(jpg_path, dest_path)

    # Print statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    for split_name, split_path in [("Train", train_path), ("Val", val_path)]:
        print(f"\n{split_name}:")
        for class_name in class_names:
            count = len(list((split_path / class_name).glob("*.jpg")))
            print(f"  {class_name}: {count}")

    return output_path


if __name__ == "__main__":
    # Update this path to your watermelon dataset
    QILIN_DIR = "/Users/danielbaez/Documents/GitHub/watermelon_eval/19_datasets"

    # Organize the dataset
    dataset_path = organize_watermelon_dataset(QILIN_DIR)
    print(f"\nDataset organized at: {dataset_path}")
