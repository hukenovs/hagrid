from typing import Dict

import torch
from torch import Tensor, nn
from torchvision import models


class ResNet(nn.Module):
    """
    Torchvision two headed ResNet and ResNext configuration
    """

    def __init__(
        self,
        num_classes: int,
        restype: str = "ResNet18",
        pretrained: bool = False,
        freezed: bool = False,
        ff: bool = False,
    ) -> None:
        """
        Torchvision two headed ResNet and ResNext configuration

        Parameters
        ----------
        num_classes : int
            Number of classes for each task
        restype : str
            Type of ResNet
        pretrained : bool
            Using pretrained weights or not
        freezed : bool
            Freezing model parameters or not
        ff : boll
            Enable full frame mode
        """

        super().__init__()

        torchvision_model = None
        self.ff = ff

        if restype == "ResNet18":
            torchvision_model = models.resnet18(pretrained=pretrained)
        elif restype == "ResNet50":
            torchvision_model = models.resnet50(pretrained=pretrained)
        elif restype == "ResNet152":
            torchvision_model = models.resnet152(pretrained=pretrained)
        elif restype == "ResNext50":
            torchvision_model = models.resnext50_32x4d(pretrained=pretrained)
        elif restype == "ResNext101":
            torchvision_model = models.resnext101_32x8d(pretrained=pretrained)

        if freezed:
            for param in torchvision_model.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(
            torchvision_model.conv1,
            torchvision_model.bn1,
            torchvision_model.relu,
            torchvision_model.maxpool,
            torchvision_model.layer1,
            torchvision_model.layer2,
            torchvision_model.layer3,
            torchvision_model.layer4,
            torchvision_model.avgpool,
        )

        num_features = torchvision_model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
        )
        if not self.ff:
            self.leading_hand = nn.Sequential(
                nn.Linear(num_features, 2),
            )

    def forward(self, img: Tensor) -> Dict:
        x = self.backbone(img)
        x = torch.flatten(x, 1)
        gesture = self.classifier(x)

        if self.ff:
            return {"gesture": gesture}
        else:
            leading_hand = self.leading_hand(x)
            return {"gesture": gesture, "leading_hand": leading_hand}
