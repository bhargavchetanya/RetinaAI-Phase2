"""
Step 1: Data Pre-processing  (APTOS 2019 - Diabetic Retinopathy, 5 grades)

Run:  python3 preprocess.py
Output: dataset/X.npy (features), dataset/y.npy (labels)
"""
import os
import cv2
import numpy as np

DATA_DIR = "dataset"
IMG_DIR = os.path.join(DATA_DIR, "train_images")
IMG_SIZE = 48          # final image is 48 x 48 x 3  ->  6912 features

GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def crop_black_border(img, tol=7):
    """Data cleaning: remove the black area around the round retina."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def ben_graham(img, sigma=10):
    """Contrast enhancement: highlights small lesions, removes uneven lighting."""
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def preprocess_image(path):
    img = cv2.imread(path)                                # 1. read
    if img is None:
        return None
    img = crop_black_border(img)                          # 2. crop
    img = cv2.resize(img, (256, 256))                     # 3. resize (same size for all)
    img = ben_graham(img)                                 # 4. enhance
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0                 # 5. normalise to [0, 1]


def main():
    import pandas as pd          # only needed here (the website imports preprocess_image without pandas)
    print("=" * 55)
    print(" DATA PRE-PROCESSING")
    print("=" * 55)

    # ---- load labels ----
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    print(f"\nTotal rows in train.csv : {len(df)}")
    print(df.head())

    # ---- basic data cleaning ----
    print(f"\nMissing values:\n{df.isnull().sum()}")
    before = len(df)
    df = df.dropna().drop_duplicates(subset="id_code")
    print(f"Removed {before - len(df)} missing/duplicate rows")

    # ---- class distribution ----
    print("\nClass distribution:")
    counts = df["diagnosis"].value_counts().sort_index()
    for g, c in counts.items():
        print(f"  {g} ({GRADES[g]:<13}): {c:5d}  ({100 * c / len(df):.1f}%)")

    # ---- process every image ----
    print(f"\nProcessing images -> {IMG_SIZE}x{IMG_SIZE} RGB ...")
    X, y, skipped = [], [], 0
    for i, row in enumerate(df.itertuples(index=False), 1):
        img = preprocess_image(os.path.join(IMG_DIR, row.id_code + ".png"))
        if img is None:
            skipped += 1
            continue
        X.append(img.flatten())                           # 6. flatten to 1-D vector
        y.append(row.diagnosis)
        if i % 250 == 0 or i == len(df):
            print(f"  {i}/{len(df)} done")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"\nSkipped (unreadable) images : {skipped}")
    print(f"Feature matrix X shape      : {X.shape}")
    print(f"Label vector  y shape       : {y.shape}")

    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)
    print("\nSaved dataset/X.npy and dataset/y.npy")


if __name__ == "__main__":
    main()
