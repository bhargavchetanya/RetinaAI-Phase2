"""
Step 2: Train the 5-layer CNN

Run:  python3 train_cnn.py                  DR detection : No DR / DR
      python3 train_cnn.py --task grade     DR grading   : all 5 grades
Needs : dataset/X.npy and dataset/y.npy  (made by preprocess.py)
Output: cnn_dr.pt or cnn_grade.pt (the trained model)
        outputs/<task>/  training_log.txt, metrics.json, training_curves.png,
                         confusion_matrix.png
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")                     # save plots as files, no window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

from cnn import RetinaCNN, get_device, predict_proba, to_tensor
from preprocess import GRADES, IMG_SIZE

# files are found next to this script, so it runs from any folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

CLASSES = {
    "dr": ["No DR", "DR"],     # DR detection: does the eye have diabetic retinopathy?
    "grade": GRADES,           # DR grading: which of the 5 grades?
}
SEED = 42

# plot colours
INK, INK_2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
TRAIN_COLOUR, VAL_COLOUR = "#2a78d6", "#eb6834"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


class Tee:
    """Print to the screen and to a log file at the same time."""
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8", buffering=1)   # write line by line
        self.screen = sys.stdout

    def write(self, text):
        self.screen.write(text)
        self.file.write(text)

    def flush(self):
        self.screen.flush()
        self.file.flush()


def duplicate_groups(X, min_corr=0.90):
    """Group number for every image - copies of the same photo get the same number.
    APTOS has some eye photos saved twice. Both copies must go to the same split,
    otherwise the CNN would be tested on a photo it was trained on (data leakage).
    Two photos are copies if their vessel patterns (green channel, centre part,
    z-scored) correlate >= min_corr; photos of different eyes correlate far less."""
    n, size = len(X), X.shape[1]
    c = size // 5                                          # leave out the rim
    f = X[:, c:size - c, c:size - c, 1].reshape(n, -1).astype(np.float32)
    f = (f - f.mean(axis=1, keepdims=True)) / (f.std(axis=1, keepdims=True) + 1e-6)
    corr = f @ f.T / f.shape[1]                            # correlation of every pair of photos
    group = np.arange(n)

    def root(a):
        while group[a] != a:
            a = group[a]
        return a

    for i, j in zip(*np.nonzero(np.triu(corr >= min_corr, k=1))):
        group[root(j)] = root(i)
    return np.array([root(i) for i in range(n)])


def augment(x):
    """Data augmentation (on the GPU): in every epoch each training image is
    randomly turned by 0/90/180/270 degrees, mirrored, shifted by up to 3 pixels
    and changed in brightness / contrast, so the CNN sees a new version of it
    each time and has to learn the lesions instead of memorising the photos.
    Only whole-pixel moves are used, so the small 48 x 48 images are not blurred."""
    n, dev, size = x.shape[0], x.device, x.shape[-1]
    turn = torch.randint(0, 4, (n,), device=dev)
    cos = torch.tensor([1.0, 0.0, -1.0, 0.0], device=dev)[turn]
    sin = torch.tensor([0.0, 1.0, 0.0, -1.0], device=dev)[turn]
    mirror = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    shift = torch.randint(-3, 4, (2, n), device=dev).float() * 2 / size
    theta = torch.stack([torch.stack([cos * mirror, -sin, shift[0]], 1),
                         torch.stack([sin * mirror, cos, shift[1]], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="nearest", align_corners=False)
    contrast = 0.8 + 0.4 * torch.rand(n, 1, 1, 1, device=dev)
    brightness = 0.4 * torch.rand(n, 1, 1, 1, device=dev) - 0.2
    return x * contrast + brightness


def predict_all(model, X, idx, mean, std, device, batch=32):
    """Class probabilities for the images X[idx]."""
    probs = []
    for s in range(0, len(idx), batch):
        x = to_tensor(X[idx[s:s + batch]], mean, std, device)
        probs.append(predict_proba(model, x).cpu())
    return torch.cat(probs).numpy()


@torch.no_grad()
def model_summary(model, device):
    """Keras-style table: output shape and number of weights of every layer."""
    model.eval()
    print(f"\n{'Layer':<28}{'Output shape':<18}{'Parameters':>12}")
    print("-" * 58)
    x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    print(f"{'input image':<28}{str(tuple(x.shape[1:])):<18}{0:>12,}")
    for i, layer in enumerate(model.features, 1):
        x = layer(x)
        params = sum(p.numel() for p in layer.parameters())
        print(f"{f'conv layer {i}':<28}{str(tuple(x.shape[1:])):<18}{params:>12,}")
    x = model.classifier[:3](x)
    print(f"{'global avg pool + dropout':<28}{str(tuple(x.shape[1:])):<18}{0:>12,}")
    dense = model.classifier[3]
    params = sum(p.numel() for p in dense.parameters())
    print(f"{'dense (output layer)':<28}{str((dense.out_features,)):<18}{params:>12,}")
    print("-" * 58)
    print(f"{'Total trainable parameters':<46}{sum(p.numel() for p in model.parameters()):>12,}")


def print_confusion(cm, classes):
    width = max(len(c) for c in classes) + 2
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(" " * width + "".join(f"{c:>{width}}" for c in classes))
    for c, row in zip(classes, cm):
        print(f"{c:<{width}}" + "".join(f"{v:>{width}}" for v in row))


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK_2)


def plot_history(history, best_epoch, path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)
    for ax, key, title, scale in [(axes[0], "loss", "Loss (lower is better)", 1),
                                  (axes[1], "acc", "Accuracy % (higher is better)", 100)]:
        ax.plot(epochs, scale * np.array(history["train_" + key]), color=TRAIN_COLOUR, lw=2,
                solid_capstyle="round", label="training images")
        ax.plot(epochs, scale * np.array(history["val_" + key]), color=VAL_COLOUR, lw=2,
                solid_capstyle="round", label="validation images")
        ax.axvline(best_epoch, color=AXIS, lw=1)
        style(ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        ax.set_xlabel("epoch", color=INK_2)
    best = 100 * history["val_acc"][best_epoch - 1]
    side = best_epoch / len(epochs)
    axes[1].plot(best_epoch, best, "o", ms=8, color=VAL_COLOUR, mec=SURFACE, mew=2, zorder=3)
    axes[1].annotate(f"kept: epoch {best_epoch}, {best:.1f}%", (best_epoch, best),
                     xytext=(0, -22), textcoords="offset points", color=INK, fontsize=9,
                     ha="left" if side < 0.25 else "right" if side > 0.75 else "center")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, labelcolor=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(cm, classes, title, path, who="the CNN"):
    n = len(classes)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)   # colour = % of each actual class
    fig, ax = plt.subplots(figsize=(1.25 * n + 2.6, 1.05 * n + 1.9), facecolor=SURFACE)
    ax.pcolormesh(share, cmap=LinearSegmentedColormap.from_list("blues", BLUES), vmin=0, vmax=1,
                  edgecolors=SURFACE, linewidth=2)
    for i in range(n):
        for j in range(n):
            ax.text(j + 0.5, i + 0.5, f"{cm[i, j]}\n{100 * share[i, j]:.0f}%", ha="center",
                    va="center", fontsize=10, color="#ffffff" if share[i, j] > 0.45 else INK)
    ax.set_xticks(np.arange(n) + 0.5, classes, rotation=20 if n > 2 else 0)
    ax.set_yticks(np.arange(n) + 0.5, classes)
    ax.invert_yaxis()
    ax.tick_params(length=0, labelcolor=INK_2)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel(f"Predicted by {who}", color=INK_2)
    ax.set_ylabel("Actual (doctor's grade)", color=INK_2)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=CLASSES, default="dr")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3, help="peak learning rate")
    args = ap.parse_args()

    out_dir = os.path.join(BASE_DIR, "outputs", args.task)
    os.makedirs(out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(out_dir, "training_log.txt"))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    classes = CLASSES[args.task]

    print("=" * 58)
    print(f" TRAINING THE 5-LAYER CNN  -  {' / '.join(classes)}")
    print("=" * 58)

    # ---- load the pre-processed images (each row of X.npy is one 48 x 48 x 3 image) ----
    X = np.load(os.path.join(DATA_DIR, "X.npy")).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    grades = np.load(os.path.join(DATA_DIR, "y.npy"))
    y = (grades > 0).astype(np.int64) if args.task == "dr" else grades
    print(f"\nImages: {X.shape[0]} x {X.shape[1]}x{X.shape[2]}x{X.shape[3]}   device: {device}")

    # ---- split: 70% train / 15% validation / 15% test (stratified by grade) ----
    # Copies of the same photo stay in one split, and the test images are used
    # only once, at the very end, so the test accuracy is honest for new patients.
    groups = duplicate_groups(X)
    ids, first, sizes = np.unique(groups, return_index=True, return_counts=True)
    print(f"Duplicate photos: {(sizes > 1).sum()} groups, {len(X) - len(ids)} extra copies"
          " (each group is kept in one split)")
    train_g, rest_g, _, rest_grades = train_test_split(ids, grades[first], test_size=0.30,
                                                       stratify=grades[first], random_state=SEED)
    val_g, test_g = train_test_split(rest_g, test_size=0.50, stratify=rest_grades, random_state=SEED)
    train_idx, val_idx, test_idx = (np.where(np.isin(groups, g))[0] for g in (train_g, val_g, test_g))
    print(f"Train: {len(train_idx)}   Validation: {len(val_idx)}   Test: {len(test_idx)}")
    for name, part in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        counts = np.bincount(y[part], minlength=len(classes))
        print(f"  {name:<6}" + "  ".join(f"{c}: {n}" for c, n in zip(classes, counts)))

    # pixel mean / std of the training images, used to standardise the input
    pixels = X[train_idx].reshape(-1, 3)
    mean, std = pixels.mean(axis=0).tolist(), pixels.std(axis=0).tolist()

    # ---- the model ----
    model = RetinaCNN(len(classes)).to(device)
    model_summary(model, device)

    steps = int(np.ceil(len(train_idx) / args.batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    epochs=args.epochs, steps_per_epoch=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    print(f"\nOptimizer: AdamW, one-cycle learning rate (peak {args.lr}), "
          f"batch size {args.batch_size}, {args.epochs} epochs")

    # ---- training ----
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best = {"acc": -1.0, "loss": np.inf}
    print(f"\n{'Epoch':>5}{'train loss':>12}{'train acc':>11}{'val loss':>10}{'val acc':>9}{'time':>7}")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        order = np.random.permutation(train_idx)
        total_loss, correct = 0.0, 0
        for s in range(0, len(order), args.batch_size):
            batch = order[s:s + args.batch_size]
            x = augment(to_tensor(X[batch], mean, std, device))
            target = torch.as_tensor(y[batch], device=device)
            logits = model(x)
            loss = loss_fn(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(batch)
            correct += (logits.argmax(1) == target).sum().item()

        prob = predict_all(model, X, val_idx, mean, std, device)
        val_loss = float(-np.log(prob[np.arange(len(val_idx)), y[val_idx]] + 1e-8).mean())
        val_acc = float((prob.argmax(1) == y[val_idx]).mean())
        history["train_loss"].append(total_loss / len(order))
        history["train_acc"].append(correct / len(order))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # keep the epoch with the best validation accuracy (ties -> lower loss)
        improved = val_acc > best["acc"] or (val_acc == best["acc"] and val_loss < best["loss"])
        if improved:
            best = {"acc": val_acc, "loss": val_loss, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        print(f"{epoch:>5}{history['train_loss'][-1]:>12.4f}{100 * history['train_acc'][-1]:>10.2f}%"
              f"{val_loss:>10.4f}{100 * val_acc:>8.2f}%{time.time() - start:>6.0f}s"
              f"{'  *' if improved else ''}")

    model.load_state_dict(best["state"])
    print(f"\n* = best so far. Kept the model from epoch {best['epoch']} "
          f"(validation accuracy {100 * best['acc']:.2f}%)")

    # ---- final evaluation on the test set ----
    prob = predict_all(model, X, test_idx, mean, std, device)
    pred, true = prob.argmax(1), y[test_idx]
    acc = float(accuracy_score(true, pred))
    cm = confusion_matrix(true, pred, labels=range(len(classes)))

    # save the trained model straight away
    model_path = os.path.join(BASE_DIR, f"cnn_{args.task}.pt")
    torch.save({"state_dict": best["state"], "task": args.task, "classes": classes,
                "img_size": IMG_SIZE, "mean": mean, "std": std,
                "test_accuracy": round(acc, 4)}, model_path)

    print("\n" + "=" * 58)
    print(f" TEST RESULTS  ({len(test_idx)} images the CNN has never seen)")
    print("=" * 58)
    print(f"\nTest accuracy : {100 * acc:.2f}%")
    metrics = {"task": args.task, "classes": classes, "model": "5-layer CNN",
               "parameters": sum(p.numel() for p in model.parameters()), "image_size": IMG_SIZE,
               "train_images": len(train_idx), "val_images": len(val_idx), "test_images": len(test_idx),
               "epochs": args.epochs, "best_epoch": best["epoch"], "val_accuracy": round(best["acc"], 4),
               "test_accuracy": round(acc, 4)}
    if args.task == "dr":
        tn, fp, fn, tp = cm.ravel()
        metrics.update(sensitivity=round(tp / (tp + fn), 4), specificity=round(tn / (tn + fp), 4),
                       precision=round(tp / (tp + fp), 4), roc_auc=round(roc_auc_score(true, prob[:, 1]), 4))
        print(f"Sensitivity   : {100 * metrics['sensitivity']:.2f}%  (eyes with DR that were found: {tp}/{tp + fn})")
        print(f"Specificity   : {100 * metrics['specificity']:.2f}%  (healthy eyes called healthy: {tn}/{tn + fp})")
        print(f"Precision     : {100 * metrics['precision']:.2f}%  (DR alarms that were right: {tp}/{tp + fp})")
        print(f"ROC-AUC       : {metrics['roc_auc']:.4f}")
        print("\nEyes with DR that were found, by the doctor's grade:")
        test_grades = grades[test_idx]
        for g in range(1, 5):
            m = test_grades == g
            print(f"  {g} {GRADES[g]:<14}: {pred[m].sum():>3}/{m.sum():<3} ({100 * pred[m].mean():.1f}%)")
    else:
        metrics.update(qwk=round(cohen_kappa_score(true, pred, weights="quadratic"), 4),
                       dr_yes_no_accuracy=round(float(((pred > 0) == (true > 0)).mean()), 4))
        print(f"Quadratic weighted kappa (QWK) : {metrics['qwk']:.4f}")
        print(f"DR yes/no accuracy (grade 0 vs 1-4): {100 * metrics['dr_yes_no_accuracy']:.2f}%")
    print("\nClassification report:")
    print(classification_report(true, pred, labels=range(len(classes)), target_names=classes,
                                digits=3, zero_division=0))
    print_confusion(cm, classes)
    metrics["confusion_matrix"] = cm.tolist()

    # ---- save the metrics and plots ----
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    plot_history(history, best["epoch"], os.path.join(out_dir, "training_curves.png"))
    plot_confusion(cm, classes, f"Test set: {len(test_idx)} unseen images, accuracy {100 * acc:.1f}%",
                   os.path.join(out_dir, "confusion_matrix.png"))
    print(f"\nSaved {os.path.basename(model_path)} and outputs/{args.task}/ "
          "(training_log.txt, metrics.json, training_curves.png, confusion_matrix.png)")


if __name__ == "__main__":
    main()
