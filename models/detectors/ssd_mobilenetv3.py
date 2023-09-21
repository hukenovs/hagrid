from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torchvision
from torch import Tensor, nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

from models.model import HaGRIDModel


class SSDLiteMobilenet_large(HaGRIDModel):
    """
    Torchvision SSDLiteMobilenet_large model for gesture detection
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        pretrained_backbone: bool = False,
        img_size: Union[int, tuple] = 320,
        img_mean: list = [0.485, 0.456, 0.406],
        img_std: list = [0.229, 0.224, 0.225],
    ):
        """
        Torchvision SSDLiteMobilenet_large model for gesture detection

        Parameters
        ----------
        num_classes: int
            Number of classes for detection
        pretrained: bool
            Using pretrained weights or not
        pretrained_backbone: bool
            Using pretrained weights for backbone or not
        img_size: int
            Size of the input image
        img_mean: list
            Mean values for the input image
        img_std: list
            Standard deviation values for the input image
        """
        super().__init__()

        if not isinstance(img_size, int):
            img_size = max(img_size)

        hagrid_model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            pretrained=pretrained, pretrained_backbone=pretrained_backbone, num_classes=num_classes
        )

        in_channels = det_utils.retrieve_out_channels(hagrid_model.backbone, (img_size, img_size))
        num_anchors = hagrid_model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)

        hagrid_model.head.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, num_classes, norm_layer
        )

        hagrid_model.transform.min_size = (img_size,)
        hagrid_model.transform.max_size = img_size

        hagrid_model.transform.image_mean = img_mean
        hagrid_model.transform.image_std = img_std

        self.hagrid_model = hagrid_model

    def __call__(self, img: Tensor, targets: Dict = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        if targets is None:
            return self.hagrid_model(img)
        else:
            model_output = self.hagrid_model(img, targets)
            return self.criterion(model_output)

    @staticmethod
    def criterion(model_output: Dict):
        loss_value = sum(loss for loss in model_output.values())
        return loss_value


class SSDLiteMobilenet_small(HaGRIDModel):
    """
    Torchvision SSDLiteMobilenet_small model for gesture detection
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        pretrained_backbone: bool = False,
        img_size: Union[int, tuple] = 320,
        img_mean: list = [0.485, 0.456, 0.406],
        img_std: list = [0.229, 0.224, 0.225],
    ):
        """
        Torchvision SSDLiteMobilenet_small model for gesture detection

        Parameters
        ----------
        num_classes: int
            Number of classes for detection
        pretrained: bool
            Using pretrained weights or not
        pretrained_backbone: bool
            Using pretrained weights for backbone or not
        img_size: int
            Size of the input image
        img_mean: list
            Mean values for the input image
        img_std: list
            Standard deviation values for the input image
        """
        super().__init__()

        if not isinstance(img_size, int):
            img_size = max(img_size)

        hagrid_model = ssdlite320_mobilenet_v3_small(
            pretrained=pretrained, pretrained_backbone=pretrained_backbone, num_classes=num_classes
        )

        in_channels = det_utils.retrieve_out_channels(hagrid_model.backbone, (img_size, img_size))
        num_anchors = hagrid_model.anchor_generator.num_anchors_per_location()
        norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)

        hagrid_model.head.classification_head = SSDLiteClassificationHead(
            in_channels, num_anchors, num_classes, norm_layer
        )

        hagrid_model.transform.min_size = (img_size,)
        hagrid_model.transform.max_size = img_size

        hagrid_model.transform.image_mean = img_mean
        hagrid_model.transform.image_std = img_std

        self.hagrid_model = hagrid_model

    def __call__(self, img: Tensor, targets: Dict = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        """
        Parameters
        ----------
        img: Tensor
            Input image
        targets: Dict
            Targets for the model

        Returns
        -------
        Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
            Model output or loss value
        """
        if targets is None:
            return self.hagrid_model(img)
        else:
            model_output = self.hagrid_model(img, targets)
            return self.criterion(model_output)

    @staticmethod
    def criterion(model_output: Dict):
        """
        Parameters
        ----------
        model_output: Dict
            Model output

        Returns
        -------
        float
            Loss value
        """
        loss_value = sum(loss for loss in model_output.values())
        return loss_value


def ssdlite320_mobilenet_v3_small(
    pretrained: bool = False,
    num_classes: int = 19,
    img_size: tuple = (320, 320),
    pretrained_backbone: bool = False,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
):
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
    anchor_generator = torchvision.models.detection.ssdlite.DefaultBoxGenerator(
        [[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95
    )
    out_channels = det_utils.retrieve_out_channels(backbone, img_size)
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
    model = SSD(
        backbone,
        anchor_generator,
        img_size,
        num_classes,
        head=torchvision.models.detection.ssdlite.SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )
    return model
