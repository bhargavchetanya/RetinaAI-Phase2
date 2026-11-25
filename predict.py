"""
Step 3: Test the trained CNN on a single retina image.

Run:  python3 predict.py <path_to_image>
e.g.  python3 predict.py samples/grade2_moderate.png
"""
import os
import sys

import pandas as pd

from cnn import load_model, predict_proba, to_tensor
from preprocess import preprocess_image, GRADES

# files are found next to this script, so it runs from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_CSV = os.path.join(BASE_DIR, "dataset", "train.csv")


def run_model(model_file, img):
    """Class probabilities from one trained CNN, or (None, None) if the file is missing."""
    path = os.path.join(BASE_DIR, model_file)
    if not os.path.exists(path):
        return None, None
    model, info = load_model(path)
    x = to_tensor(img[None], info["mean"], info["std"], "cpu")
    return predict_proba(model, x)[0].tolist(), info


def show(title, probs, info):
    pred = probs.index(max(probs))
    print(f"\n{title}  (5-layer CNN, {100 * info['test_accuracy']:.1f}% accurate on unseen test images)")
    print(f"Prediction : {info['classes'][pred]}")
    print("Probabilities:")
    for name, p in zip(info["classes"], probs):
        print(f"  {name:<14}: {100 * p:5.1f}%")
    return pred


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 predict.py <path_to_image>")
        sys.exit(1)
    path = sys.argv[1]

    # same pre-processing as training
    img = preprocess_image(path)
    if img is None:
        print("Could not read image:", path)
        sys.exit(1)

    print(f"\nImage      : {path}")
    probs, info = run_model("cnn_dr.pt", img)
    if probs is None:
        print("cnn_dr.pt not found - train it first:  python3 train_cnn.py")
        sys.exit(1)
    has_dr = show("DR DETECTION", probs, info) == 1
    probs, info = run_model("cnn_grade.pt", img)      # optional 5-grade model
    if probs is not None:
        show("DR GRADE", probs, info)

    # if the image is from our dataset, also show the true grade
    # (skipped when the dataset is not present, e.g. on a shared copy)
    if not os.path.exists(LABELS_CSV):
        return
    img_id = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(LABELS_CSV)
    row = df[df["id_code"] == img_id]
    if len(row):
        actual = int(row["diagnosis"].iloc[0])
        print(f"\nActual     : {actual} ({GRADES[actual]})  ->  DR detection "
              f"{'CORRECT' if (actual > 0) == has_dr else 'WRONG'}")


if __name__ == "__main__":
    main()
