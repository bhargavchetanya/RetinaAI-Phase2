"""
The 4 pre-trained models (transfer learning). Every one of them first learned to see
on ImageNet (1.2 million everyday photos); its old 1000-class output layer is replaced
by a new output layer for our classes, and then all layers are fine-tuned on the
retina photos. Used by train_pretrained.py, predict_pretrained.py and app.py.

    model               idea                                              weights   download
    ResNet-18           residual blocks: skip connections add a block's    11.2 M    45 MB
                        input to its output
    EfficientNet-B0     MBConv blocks (depthwise conv + squeeze-and-         4.0 M    21 MB
                        excitation), depth / width / resolution scaled
                        together
    DenseNet-121        dense blocks: every layer gets the feature maps      7.0 M    31 MB
                        of ALL earlier layers in its block
    MobileNetV3-Large   light inverted-residual blocks with hard-swish,      3.0 M    21 MB
                        designed for phones

All 4 use the same input as the 5-layer CNN: the 48 x 48 photo from preprocess.py,
enlarged to 128 x 128 (bilinear) and turned from BGR (OpenCV) to RGB (ImageNet).
    enlarge -> pre-trained feature layers -> global average pooling -> Dropout -> Dense
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from resnet import ENLARGED_SIZE, Enlarge, RetinaResNet

# key -> settings. The key is also the start of the file names: <key>_dr.pt, outputs/<key>_dr/
ARCHS = {
    "resnet18": {
        "name": "ResNet-18",
        "lr": 5e-4,
        "download": "45 MB",
    },
    "efficientnet_b0": {
        "name": "EfficientNet-B0",
        "build": models.efficientnet_b0,
        "weights": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        "features_out": 1280,
        "lr": 1e-3,
        "download": "21 MB",
    },
    "densenet121": {
        "name": "DenseNet-121",
        "build": models.densenet121,
        "weights": models.DenseNet121_Weights.IMAGENET1K_V1,
        "features_out": 1024,
        "relu_after_features": True,     # torchvision's DenseNet applies ReLU after its last BatchNorm
        "lr": 5e-4,
        "download": "31 MB",
    },
    "mobilenet_v3_large": {
        "name": "MobileNetV3-Large",
        "build": models.mobilenet_v3_large,
        "weights": models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        "features_out": 960,
        "lr": 1e-3,
        "download": "21 MB",
    },
}


class RetinaPretrained(nn.Module):
    """EfficientNet-B0 / DenseNet-121 / MobileNetV3-Large with a new output layer."""
    def __init__(self, arch, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        cfg = ARCHS[arch]
        net = cfg["build"](weights=cfg["weights"] if pretrained else None)   # first run downloads the weights
        self.arch = arch
        self.enlarge = Enlarge()
        self.features = net.features                     # the pre-trained feature layers
        head = [nn.ReLU(inplace=True)] if cfg.get("relu_after_features") else []
        self.classifier = nn.Sequential(*head,           # new output layer for our classes
                                        nn.AdaptiveAvgPool2d(1),
                                        nn.Flatten(),
                                        nn.Dropout(dropout),
                                        nn.Linear(cfg["features_out"], num_classes))

    def forward(self, x):
        return self.classifier(self.features(self.enlarge(x)))

    def stages(self):
        """(name, part) of every stage, for the summary table in train_pretrained.py."""
        rows = [(f"enlarge to {ENLARGED_SIZE}x{ENLARGED_SIZE}", self.enlarge)]
        rows += [(f"features.{n}: {type(m).__name__}", m) for n, m in self.features.named_children()]
        rows += [("global avg pool + dropout", self.classifier[:-1]),
                 ("dense (output layer)", self.classifier[-1])]
        return rows


def build_model(arch, num_classes, pretrained=True):
    if arch == "resnet18":
        return RetinaResNet(num_classes, pretrained=pretrained)      # resnet.py (kept so old .pt files load)
    return RetinaPretrained(arch, num_classes, pretrained=pretrained)


def load_pretrained(path, device="cpu"):
    """Load a model saved by train_pretrained.py (or the older train_resnet.py).
    Returns (model, checkpoint info)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    arch = ckpt.get("arch", "resnet18")                 # files from train_resnet.py have no "arch"
    model = build_model(arch, len(ckpt["classes"]), pretrained=False)   # the weights come from the file
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
