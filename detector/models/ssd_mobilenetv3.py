import warnings
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Optional, Union, Tuple, Dict, List, Iterator

import torch
import torchvision
from torch import nn, Tensor
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from detector.models.model import TorchVisionModel


class SSDLiteMobilenet_large(TorchVisionModel):
    """
    Torchvision SSDLite model for gesture detection
    """

    def __init__(self, num_classes: int, pretrained: bool = False):
        """
        Torchvision SSDLite model for gesture detection

        Parameters
        ----------
        num_classes: int
            Number of classes for detection
        """
        super().__init__()
        self.num_classes = num_classes

        torchvision_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=pretrained)

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

    def load_state_dict(self, checkpoint):
        self.torchvision_model.load_state_dict(checkpoint)

    def state_dict(self) -> OrderedDict:
        return self.torchvision_model.state_dict()


def SSDLiteMobilenet_small(pretrained: bool = False,
                              num_classes: int = 19,
                              pretrained_backbone: bool = False,
                              trainable_backbone_layers: Optional[int] = None,
                              **kwargs: Any, ):
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = torchvision.models.detection.ssdlite._validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 6
    )

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    reduce_tail = not pretrained_backbone

    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = torchvision.models.mobilenet_v3_small(
        pretrained=pretrained_backbone, progress=True, norm_layer=norm_layer, reduced_tail=reduce_tail, **kwargs
    )
    if not pretrained_backbone:
        # Change the default initialization scheme if not pretrained
        torchvision.models.detection.ssdlite._normal_init(backbone)
    backbone = torchvision.models.detection.ssdlite._mobilenet_extractor(
        backbone,
        trainable_backbone_layers,
        norm_layer,
    )

    size = (320, 320)
    anchor_generator = torchvision.models.detection.ssdlite.DefaultBoxGenerator([[2, 3] for _ in range(6)],
                                                                                min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    assert len(out_channels) == len(anchor_generator.aspect_ratios)

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, 1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    kwargs = {**defaults, **kwargs}
    model = torchvision.models.detection.SSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=torchvision.models.detection.ssdlite.SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )
    return model
