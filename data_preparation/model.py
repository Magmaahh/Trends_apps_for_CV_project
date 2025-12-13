import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class MouthEmbeddingResNet3D(nn.Module):
    """
    3D ResNet adapted for grayscale mouth video clips.
    Outputs a fixed-dimensional embedding.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.backbone = r3d_18(weights="R3D_18_Weights.DEFAULT")

        old_conv = self.backbone.stem[0]
        new_conv = nn.Conv3d(
            1, 64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )

        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.sum(dim=1, keepdim=True)

        self.backbone.stem[0] = new_conv
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            embedding_dim
        )

    def forward(self, x):
        return self.backbone(x)