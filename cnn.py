"""
The 5-layer CNN (Convolutional Neural Network) used by train_cnn.py and predict.py.

    input: retina photo pre-processed by preprocess.py, 3 x 48 x 48
      conv layer 1 :  32 filters 3x3 -> BatchNorm -> ReLU            ->  32 x 48 x 48
      conv layer 2 :  64 filters 3x3 -> BatchNorm -> ReLU -> MaxPool ->  64 x 24 x 24
      conv layer 3 : 128 filters 3x3 -> BatchNorm -> ReLU -> MaxPool -> 128 x 12 x 12
      conv layer 4 : 256 filters 3x3 -> BatchNorm -> ReLU -> MaxPool -> 256 x  6 x  6
      conv layer 5 : 512 filters 3x3 -> BatchNorm -> ReLU -> MaxPool -> 512 x  3 x  3
      global average pooling -> 512 numbers -> Dropout -> Dense (fully connected) -> classes
"""
import torch
import torch.nn as nn

CHANNELS = [32, 64, 128, 256, 512]     # filters in conv layers 1..5


def conv_layer(c_in, c_out, pool):
    """One convolutional layer: 3x3 convolution -> batch norm -> ReLU (-> 2x2 max-pool)."""
    layers = [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
              nn.BatchNorm2d(c_out),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class RetinaCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        super().__init__()
        layers, c_in = [], 3
        for i, c_out in enumerate(CHANNELS):
            # no pooling after layer 1: a 48 x 48 image is already small
            layers.append(conv_layer(c_in, c_out, pool=i > 0))
            c_in = c_out
        self.features = nn.Sequential(*layers)          # the 5 conv layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                     # 512 x 3 x 3 -> 512 x 1 x 1
            nn.Flatten(),
            nn.Dropout(dropout),                         # regularisation
            nn.Linear(c_in, num_classes),                # output layer
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_device():
    """Apple GPU (MPS) > NVIDIA GPU (CUDA) > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensor(images, mean, std, device):
    """Images (N, H, W, 3) with values 0-1 -> standardised float tensor (N, 3, H, W) on the device."""
    x = torch.as_tensor(images, dtype=torch.float32).to(device).permute(0, 3, 1, 2)
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def predict_proba(model, x):
    """Class probabilities, averaged over 4 mirrored copies of each image
    (test-time augmentation: the same eye flipped should get the same answer)."""
    model.eval()
    p = torch.softmax(model(x), dim=1)
    for dims in ([3], [2], [2, 3]):
        p += torch.softmax(model(x.flip(dims)), dim=1)
    return p / 4


def load_model(path, device="cpu"):
    """Load a trained model saved by train_cnn.py. Returns (model, checkpoint info)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = RetinaCNN(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
