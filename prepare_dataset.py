"""Regenerate the included sklearn handwritten digits dataset and CSV."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

digits = load_digits()
images = digits.images.astype(np.uint8)
labels = digits.target.astype(np.uint8)
np.savez_compressed(DATA_DIR / "digits_dataset.npz", images=images, labels=labels)
flat = images.reshape(len(images), -1)
df = pd.DataFrame(flat, columns=[f"pixel_{i}" for i in range(64)])
df["label"] = labels
df.to_csv(DATA_DIR / "digits_dataset.csv", index=False)
print(f"Dataset saved: {len(images)} images, shape 8x8, labels 0-9")
print(f"NPZ: {DATA_DIR / 'digits_dataset.npz'}")
print(f"CSV: {DATA_DIR / 'digits_dataset.csv'}")
