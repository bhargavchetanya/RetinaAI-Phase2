"""
Step 3 (Model 2): Test the pre-trained ResNet-18 on a single retina image.

Run:  python3 predict_resnet.py <path_to_image>
e.g.  python3 predict_resnet.py samples/grade2_moderate.png
Uses resnet18_dr.pt (No DR / DR) and resnet18_grade.pt (5 grades) - whichever are trained.
"""
import os
import sys

import pandas as pd

from cnn import predict_proba, to_tensor
from preprocess import preprocess_image, GRADES
from resnet import load_resnet

# files are found next to this script, so it runs from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS_CSV = os.path.join(BASE_DIR, "dataset", "train.csv")


def run_model(model_file, img):
    """Class probabilities from one trained ResNet, or (None, None) if the file is missing."""
    path = os.path.join(BASE_DIR, model_file)
    if not os.path.exists(path):
        return None, None
    model, info = load_resnet(path)
    x = to_tensor(img[None], info["mean"], info["std"], "cpu")
    return predict_proba(model, x)[0].tolist(), info


def show(title, probs, info):
    pred = probs.index(max(probs))
    print(f"\n{title}  (ResNet-18 pre-trained, {100 * info['test_accuracy']:.1f}% accurate "
          "on unseen test images)")
    print(f"Prediction : {info['classes'][pred]}")
    print("Probabilities:")
    for name, p in zip(info["classes"], probs):
        print(f"  {name:<14}: {100 * p:5.1f}%")
    return pred


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 predict_resnet.py <path_to_image>")
        sys.exit(1)
    path = sys.argv[1]

    # same pre-processing as training
    img = preprocess_image(path)
    if img is None:
        print("Could not read image:", path)
        sys.exit(1)

    print(f"\nImage      : {path}")
    has_dr = grade = None
    probs, info = run_model("resnet18_dr.pt", img)
    if probs is not None:
        has_dr = show("DR DETECTION", probs, info) == 1
    probs, info = run_model("resnet18_grade.pt", img)
    if probs is not None:
        grade = show("DR GRADE", probs, info)
    if has_dr is None and grade is None:
        print("No trained ResNet found - train it first:  python3 train_resnet.py")
        sys.exit(1)

    # if the image is from our dataset, also show the true grade
    # (skipped when the dataset is not present, e.g. on a shared copy)
    if not os.path.exists(LABELS_CSV):
        return
    img_id = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(LABELS_CSV)
    row = df[df["id_code"] == img_id]
    if len(row):
        actual = int(row["diagnosis"].iloc[0])
        checks = []
        if has_dr is not None:
            checks.append(f"DR detection {'CORRECT' if (actual > 0) == has_dr else 'WRONG'}")
        if grade is not None:
            checks.append(f"grade {'CORRECT' if actual == grade else 'WRONG'}")
        print(f"\nActual     : {actual} ({GRADES[actual]})  ->  " + ", ".join(checks))


if __name__ == "__main__":
    main()
