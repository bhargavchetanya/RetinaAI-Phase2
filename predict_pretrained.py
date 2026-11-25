"""
Step 3 (Models 2-5): Test all 4 pre-trained models on a single retina image.

Run:  python3 predict_pretrained.py <path_to_image>
e.g.  python3 predict_pretrained.py samples/grade2_moderate.png
Uses every trained file among  <model>_dr.pt / <model>_grade.pt  for
resnet18, efficientnet_b0, densenet121 and mobilenet_v3_large.
"""
import os
import sys

import pandas as pd

from cnn import predict_proba, to_tensor
from preprocess import GRADES, preprocess_image
from pretrained import ARCHS, load_pretrained

# files are found next to this script, so it runs from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_CSV = os.path.join(BASE_DIR, "dataset", "train.csv")
TASKS = {"dr": "DR DETECTION (No DR / DR)", "grade": "DR GRADE (5 classes)"}


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 predict_pretrained.py <path_to_image>")
        sys.exit(1)
    path = sys.argv[1]

    img = preprocess_image(path)                    # same pre-processing as training
    if img is None:
        print("Could not read image:", path)
        sys.exit(1)

    # if the image is from our dataset, also show the true grade
    actual = None
    if os.path.exists(LABELS_CSV):
        df = pd.read_csv(LABELS_CSV)
        row = df[df["id_code"] == os.path.splitext(os.path.basename(path))[0]]
        if len(row):
            actual = int(row["diagnosis"].iloc[0])

    print(f"\nImage      : {path}")
    if actual is not None:
        print(f"Actual     : {actual} ({GRADES[actual]})")
    found = False
    for task, title in TASKS.items():
        rows = []
        for key, cfg in ARCHS.items():
            model_path = os.path.join(BASE_DIR, f"{key}_{task}.pt")
            if not os.path.exists(model_path):
                continue
            model, info = load_pretrained(model_path)
            x = to_tensor(img[None], info["mean"], info["std"], "cpu")
            probs = predict_proba(model, x)[0].tolist()
            rows.append((cfg["name"], info, probs))
        if not rows:
            continue
        found = True
        classes = rows[0][1]["classes"]
        print(f"\n{title}")
        print(f"  {'Model':<20}{'test acc':>9}  {'Prediction':<15}" + "".join(f"{c:>15}" for c in classes))
        for name, info, probs in rows:
            pred = probs.index(max(probs))
            mark = ""
            if actual is not None:
                right = (pred == 1) == (actual > 0) if task == "dr" else pred == actual
                mark = "  CORRECT" if right else "  WRONG"
            print(f"  {name:<20}{100 * info['test_accuracy']:>8.1f}%  {classes[pred]:<15}"
                  + "".join(f"{100 * p:>14.1f}%" for p in probs) + mark)
    if not found:
        print("No trained pre-trained model found - train one first, e.g.\n"
              "  python3 train_pretrained.py --model efficientnet_b0 --task dr")
        sys.exit(1)


if __name__ == "__main__":
    main()
