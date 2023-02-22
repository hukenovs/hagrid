from typing import Dict

import torchvision
from torch import Tensor, nn


class MobileNetV3(nn.Module):
    """
    Torchvision two headed MobileNet V3 configuration
    """

    def __init__(self, num_classes: int, size: str = "large", pretrained: bool = False, freezed: bool = False,
                 ff: bool = False) -> None:
        """
        Torchvision two headed MobileNet V3 configuration

        Parameters
        ----------
        num_classes : int
            Number of classes for each task
        size : str
            Size of MobileNetV3 ('small' or 'large')
        pretrained : bool
            Using pretrained weights or not
        freezed : bool
            Freezing model parameters or not
        ff : boll
            Enable full frame mode
        """

        super(MobileNetV3, self).__init__()
        self.ff = ff

        if size == "small":
            torchvision_model = torchvision.models.mobilenet_v3_small(pretrained)
            in_features = 576
            out_features = 1024
        else:
            torchvision_model = torchvision.models.mobilenet_v3_large(pretrained)
            in_features = 960
            out_features = 1280

        if freezed:
            for param in torchvision_model.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(torchvision_model.features, torchvision_model.avgpool)

        self.gesture_classifier = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=out_features, out_features=num_classes),
        )
        if not self.ff:
            self.leading_hand_classifier = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.Hardswish(),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=out_features, out_features=2),
            )

        self.num_classes = num_classes

    def forward(self, x: Tensor) -> Dict:
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        gesture = self.gesture_classifier(x)

        if self.ff:
            return {"gesture": gesture}
        else:
            leading_hand = self.leading_hand_classifier(x)
            return {"gesture": gesture, "leading_hand": leading_hand}
