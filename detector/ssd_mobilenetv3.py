import torch
import torchvision
from torch import nn, Tensor
from collections import OrderedDict
from functools import partial
from typing import Tuple, Dict, List, Iterator
from detector.model import TorchVisionModel
from torchvision.models.detection import _utils as det_utils  # NOQA
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead  # NOQA


class SSDMobilenet(TorchVisionModel):
    """
    Torchvision SSDLite model for gesture detection
    """
    def __init__(self, num_classes: int):
        """
        Torchvision SSDLite model for gesture detection

        Parameters
        ----------
        num_classes: int
            Number of classes for detection
        """
        super().__init__()
        self.num_classes = num_classes

        torchvision_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=False)

        in_channels = det_utils.retrieve_out_channels(torchvision_model.backbone, (320, 320))
        num_anchors = torchvision_model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)

        torchvision_model.head.classification_head = SSDLiteClassificationHead(
            in_channels,
            num_anchors,
            num_classes,
            norm_layer
        )

        self.torchvision_model = torchvision_model

    def __call__(self, img: Tensor, targets: Dict = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if targets is None:
            return self.torchvision_model(img)
        else:
            return self.torchvision_model(img, targets)

    @staticmethod
    def criterion(model_output: Dict):
        loss_value = sum(loss for loss in model_output.values())
        return loss_value

    def to(self, device: str):
        self.torchvision_model.to(device)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.torchvision_model.parameters()

    def train(self) -> nn.Module:
        return self.torchvision_model.train()

    def eval(self) -> nn.Module:
        return self.torchvision_model.eval()

    def load_state_dict(self, checkpoint_path: str, map_location : str = None):
        self.torchvision_model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))

    def state_dict(self) -> OrderedDict:
        return self.torchvision_model.state_dict()
