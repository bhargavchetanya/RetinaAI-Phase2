"""
Shortcut kept from before: trains ResNet-18 with train_pretrained.py.

Run:  python3 train_resnet.py                  = python3 train_pretrained.py --model resnet18 --task grade
      python3 train_resnet.py --task dr        = python3 train_pretrained.py --model resnet18 --task dr
"""
import sys

from train_pretrained import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--model", "resnet18"]
    if "--task" not in sys.argv:
        sys.argv += ["--task", "grade"]
    main()
