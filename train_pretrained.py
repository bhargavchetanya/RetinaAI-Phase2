"""
Step 2 (Models 2-5): Train one of the 4 models pre-trained on ImageNet (transfer learning)

Run:  python3 train_pretrained.py --model efficientnet_b0 --task dr      DR detection : No DR / DR
      python3 train_pretrained.py --model efficientnet_b0 --task grade   DR grading   : all 5 grades
      --model  resnet18 | efficientnet_b0 | densenet121 | mobilenet_v3_large
Needs : dataset/X.npy and dataset/y.npy  (made by preprocess.py), and internet on
        the first run of each model to download its ImageNet weights (21-45 MB)
Output: <model>_<task>.pt (the trained model), e.g. efficientnet_b0_dr.pt
        outputs/<model>_<task>/  training_log.txt, metrics.json, training_curves.png,
                                 confusion_matrix.png
It uses the same data split, augmentation and test metrics as train_cnn.py, so all
models are tested on exactly the same photos.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

from cnn import get_device, to_tensor
from preprocess import GRADES, IMG_SIZE
from pretrained import ARCHS, build_model
from train_cnn import (BASE_DIR, CLASSES, DATA_DIR, SEED, Tee, augment, duplicate_groups,
                       plot_confusion, plot_history, predict_all, print_confusion)


@torch.no_grad()
def model_summary(model, device):
    """Keras-style table: output shape and number of weights of every stage."""
    model.eval()
    print(f"\n{'Layer':<40}{'Output shape':<18}{'Parameters':>12}")
    print("-" * 70)
    x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    print(f"{'input image':<40}{str(tuple(x.shape[1:])):<18}{0:>12,}")
    for name, part in model.stages():
        x = part(x)
        params = sum(p.numel() for p in part.parameters())
        print(f"{name:<40}{str(tuple(x.shape[1:])):<18}{params:>12,}")
    print("-" * 70)
    print(f"{'Total trainable parameters':<58}{sum(p.numel() for p in model.parameters()):>12,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=ARCHS, required=True)
    ap.add_argument("--task", choices=CLASSES, default="grade")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None,
                    help="peak learning rate (default depends on the model; lower than the CNN's "
                         "because pre-trained weights only need fine-tuning)")
    args = ap.parse_args()
    arch = ARCHS[args.model]
    model_name = arch["name"]
    if args.lr is None:
        args.lr = arch["lr"]

    run = f"{args.model}_{args.task}"
    out_dir = os.path.join(BASE_DIR, "outputs", run)
    os.makedirs(out_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(out_dir, "training_log.txt"))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    classes = CLASSES[args.task]

    print("=" * 70)
    print(f" TRAINING {model_name.upper()} (PRE-TRAINED ON IMAGENET)  -  {' / '.join(classes)}")
    print("=" * 70)

    # ---- load the pre-processed images (each row of X.npy is one 48 x 48 x 3 image) ----
    X = np.load(os.path.join(DATA_DIR, "X.npy")).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    grades = np.load(os.path.join(DATA_DIR, "y.npy"))
    y = (grades > 0).astype(np.int64) if args.task == "dr" else grades
    print(f"\nImages: {X.shape[0]} x {X.shape[1]}x{X.shape[2]}x{X.shape[3]}   device: {device}")

    # ---- split: 70% train / 15% validation / 15% test - the same split as train_cnn.py ----
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

    # ---- the model: ImageNet weights and a new output layer ----
    print(f"\nLoading {model_name} with ImageNet weights (first run downloads {arch['download']}) ...")
    model = build_model(args.model, len(classes)).to(device)
    model_summary(model, device)

    steps = int(np.ceil(len(train_idx) / args.batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    epochs=args.epochs, steps_per_epoch=steps)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    print(f"\nOptimizer: AdamW, one-cycle learning rate (peak {args.lr}), "
          f"batch size {args.batch_size}, {args.epochs} epochs, all layers fine-tuned")

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
    model_path = os.path.join(BASE_DIR, f"{run}.pt")
    torch.save({"state_dict": best["state"], "arch": args.model, "name": model_name,
                "task": args.task, "classes": classes,
                "img_size": IMG_SIZE, "mean": mean, "std": std,
                "test_accuracy": round(acc, 4)}, model_path)

    print("\n" + "=" * 70)
    print(f" TEST RESULTS  ({len(test_idx)} images {model_name} has never seen)")
    print("=" * 70)
    print(f"\nTest accuracy : {100 * acc:.2f}%")
    metrics = {"task": args.task, "classes": classes, "model": f"{model_name} pre-trained on ImageNet",
               "arch": args.model,
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
    plot_confusion(cm, classes, f"{model_name}, test set: {len(test_idx)} unseen images, accuracy {100 * acc:.1f}%",
                   os.path.join(out_dir, "confusion_matrix.png"), who=model_name)
    print(f"\nSaved {run}.pt and outputs/{run}/ "
          "(training_log.txt, metrics.json, training_curves.png, confusion_matrix.png)")


if __name__ == "__main__":
    main()
