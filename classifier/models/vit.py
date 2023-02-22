from typing import Dict

import torch
import torchvision
from torch import Tensor, nn


class Vit(nn.Module):
    """
    Torchvision two headed Vision Transformer configuration
    for gesture and leading hand classification tasks
    """

    def __init__(self, num_classes: int, pretrained: bool = False, freezed: bool = False, ff=False) -> None:
        """
        Torchvision two headed Vision Transformer configuration
        for gesture and leading hand classification tasks

        Parameters
        ----------
        num_classes : int
            Number of classes for each task
        pretrained : bool
            Using pretrained weights or not
        freezed : bool
            Freezing model parameters or not
        ff : bool
            Enable full frame mode
        """
        super(Vit, self).__init__()
        self.ff = ff
        self.torchvision_model = torchvision.models.vit_b_32(pretrained)

        if freezed:
            for param in self.torchvision_model.parameters():
                param.requires_grad = False

        self.backbone = nn.Sequential(self.torchvision_model.encoder)

        self.gesture_classifier = nn.Sequential(
            nn.Linear(in_features=self.torchvision_model.hidden_dim, out_features=num_classes)
        )
        if not self.ff:
            self.leading_hand_classifier = nn.Sequential(
                nn.Linear(in_features=self.torchvision_model.hidden_dim, out_features=2)
            )

    def forward(self, x: Tensor) -> Dict:
        x = self.torchvision_model._process_input(x)  # NOQA
        n = x.shape[0]

        batch_class_token = self.torchvision_model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.backbone(x)

        x = x[:, 0]

        gesture = self.gesture_classifier(x)

        if self.ff:
            return {"gesture": gesture}
        else:
            leading_hand = self.leading_hand_classifier(x)

            return {"gesture": gesture, "leading_hand": leading_hand}
