"""
Model 2: ResNet-18 pre-trained on ImageNet (1.2 million photos), fine-tuned on the
retina photos (transfer learning). Used by train_resnet.py and predict_resnet.py.

    input: retina photo pre-processed by preprocess.py, 3 x 48 x 48
      enlarge to 128 x 128 (bilinear), BGR -> RGB             ->   3 x 128 x 128
      stem    : conv 7x7 (64 filters) + BatchNorm + ReLU + MaxPool ->  64 x  32 x  32
      layer 1 : 2 residual blocks,  64 filters                 ->  64 x  32 x  32
      layer 2 : 2 residual blocks, 128 filters                 -> 128 x  16 x  16
      layer 3 : 2 residual blocks, 256 filters                 -> 256 x   8 x   8
      layer 4 : 2 residual blocks, 512 filters                 -> 512 x   4 x   4
      global average pooling -> 512 numbers -> Dropout -> Dense (new output layer) -> classes

A residual block is two 3x3 conv layers plus a skip connection that adds the
block's input to its output, so very deep networks still train well.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

ENLARGED_SIZE = 128


class Enlarge(nn.Module):
    """48 x 48 -> 128 x 128 (bilinear) and BGR -> RGB: the pre-trained filters were
    learned on larger RGB photos, while OpenCV reads images as BGR."""
    def forward(self, x):
        return F.interpolate(x.flip(1), size=(ENLARGED_SIZE, ENLARGED_SIZE), mode="bilinear",
                             align_corners=False)


class RetinaResNet(nn.Module):
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)      # first run downloads the ImageNet weights (45 MB)
        self.enlarge = Enlarge()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.classifier = nn.Sequential(             # new output layer for our classes
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(net.fc.in_features, num_classes),
        )

    def forward(self, x):
        x = self.stem(self.enlarge(x))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.classifier(x)

    def stages(self):
        """(name, part) of every stage, for the summary table in train_resnet.py."""
        return [(f"enlarge to {ENLARGED_SIZE}x{ENLARGED_SIZE}", self.enlarge),
                ("stem: conv 7x7 + BN + ReLU + max-pool", self.stem),
                ("layer 1: 2 residual blocks", self.layer1),
                ("layer 2: 2 residual blocks", self.layer2),
                ("layer 3: 2 residual blocks", self.layer3),
                ("layer 4: 2 residual blocks", self.layer4),
                ("global avg pool + dropout", self.classifier[:3]),
                ("dense (output layer)", self.classifier[3])]


def load_resnet(path, device="cpu"):
    """Load a model saved by train_resnet.py. Returns (model, checkpoint info)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = RetinaResNet(len(ckpt["classes"]), pretrained=False)    # the weights come from the file
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
