from collections import OrderedDict
from typing import Dict

import torchvision
from torch import Tensor, nn


def VitB32(**kwargs):
    """
    Torchvision two headed Vision Transformer configuration with patch size 32
    """
    return Vit(patch_size=32, **kwargs)


def VitB16(**kwargs):
    """
    Torchvision two headed Vision Transformer configuration with patch size 16
    """
    return Vit(patch_size=16, **kwargs)


class Vit(nn.Module):
    """
    Torchvision two headed Vision Transformer configuration
    for gesture and leading hand classification tasks
    """

    def __init__(self, num_classes: int, pretrained: bool = False, patch_size=16, *args, **kwargs) -> None:
        """
        Torchvision two headed Vision Transformer configuration
        for gesture and leading hand classification tasks

        Parameters
        ----------
        num_classes : int
            Number of classes for each task
        pretrained : bool
            Using pretrained weights or not
        patch_size : int
            Patch size for the transformer
        """
        super().__init__()
        self.hagrid_model = getattr(torchvision.models, f"vit_b_{patch_size}")(pretrained)

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(in_features=self.hagrid_model.hidden_dim, out_features=num_classes)

        self.hagrid_model.heads = nn.Sequential(heads_layers)

        nn.init.zeros_(self.hagrid_model.heads.head.weight)
        nn.init.zeros_(self.hagrid_model.heads.head.bias)

    def forward(self, x: Tensor) -> Dict:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor

        Returns
        -------
        Dict
            Dictionary containing the model output
        """
        gesture = self.hagrid_model(x)

        return gesture
