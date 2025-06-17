import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

# Generate K splits for KAIST RGB-T dataset

K = 5

"""
[Split 1] Train: 10559 images, Val: 1979 images
[Split 2] Train: 9905 images, Val: 2633 images
[Split 3] Train: 9806 images, Val: 2732 images
[Split 4] Train: 9520 images, Val: 3018 images
[Split 5] Train: 10362 images, Val: 2176 images
"""

# Original txt file (fixed)
orig_txt = Path("datasets/kaist-rgbt/train-all-04.txt")
orig_test_txt = Path("datasets/kaist-rgbt/test-all-20.txt")

# Read original list
with open(orig_txt, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

# Group by set##_V### (used as key for each fold)
grouped = defaultdict(list)
for line in lines:
    filename = Path(line).name
    key = "_".join(filename.split("_")[:2])  # e.g., set02_V004
    grouped[key].append(line)

# All unique keys
all_keys = list(grouped.keys())
random.shuffle(all_keys)

# Divide into K folds
folds = [[] for _ in range(K)]
for i, key in enumerate(all_keys):
    folds[i % K].append(key)

# Create each split
for split_index in range(1, K + 1):
    output_base = Path(f"datasets/kaist-rgbt-split{split_index}")
    os.makedirs(output_base, exist_ok=True)

    val_keys = folds[split_index - 1]
    train_keys = [k for i, fold in enumerate(folds) if i != (split_index - 1) for k in fold]

    train_lines = [l for k in train_keys for l in grouped[k]]
    val_lines = [l for k in val_keys for l in grouped[k]]

    # Replace path prefix: datasets/kaist-rgbt → datasets/kaist-rgbt-split{index}
    train_lines = [l.replace("datasets/kaist-rgbt", f"datasets/kaist-rgbt-split{split_index}") for l in train_lines]
    val_lines = [l.replace("datasets/kaist-rgbt", f"datasets/kaist-rgbt-split{split_index}") for l in val_lines]

    # Save train/val txt files
    with open(output_base / "train-all-04.txt", "w") as f:
        f.write("\n".join(train_lines))
    with open(output_base / "val-all-04.txt", "w") as f:
        f.write("\n".join(val_lines))

    # Save test txt file
    with open(orig_test_txt, "r") as f:
        test_lines = [line.strip() for line in f if line.strip()]
        test_lines = [l.replace("datasets/kaist-rgbt", f"datasets/kaist-rgbt-split{split_index}") for l in test_lines]
    with open(output_base / "test-all-20.txt", "w") as f:
        f.write("\n".join(test_lines))

    # Create symlinks for 'train' and 'test' folders
    for sub in ["train", "test"]:
        src = (Path("datasets/kaist-rgbt") / sub).resolve()
        dst = output_base / sub
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)

    # Print stats
    print(f"[Split {split_index}] Train: {len(train_lines)} images, Val: {len(val_lines)} images")
