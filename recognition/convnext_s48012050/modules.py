"""
Contains the source code for the components of the model/s
"""
import torch
from torch import nn

###### BASELINE CNN ######
class ConvBlock(nn.Module):
    """Two convolution layers with batch normalisation and relu activation function, followed by
    a final max pool"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """A CNN baseline for AD vs NC classification.
    Uses four ConvBlocks, then average pooling and linear layers,
    somewhat mimicking the ConvNeXt architecture.

    Input: (batch, 1, H, W)
    Output: (batch, num_classes) in raw logits
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


###### CONVNEXT ######
class ConvNext():
    def __init__(self):
        raise(NotImplementedError("Convnext is not yet implemented"))