from functools import partial
from typing import Union

import torch
import torchvision
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

from models.model import HaGRIDModel


class RetinaNet_ResNet50(HaGRIDModel):
    """
    Torchvision SSD model for gesture detection
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        pretrained_backbone: bool = False,
        img_size: Union[int, tuple] = 300,
        img_mean: list = None,
        img_std: list = None,
    ):
        """
        Torchvision SSDLite model for gesture detection

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

        hagrid_model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            pretrained=pretrained, pretrained_backbone=pretrained_backbone, num_classes=num_classes
        )

        # in_features = hagrid_model.backbone.out_channels
        num_anchors = hagrid_model.head.classification_head.num_anchors
        norm_layer = partial(torch.nn.GroupNorm, 32)

        hagrid_model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256, num_anchors=num_anchors, num_classes=num_classes, norm_layer=norm_layer
        )

        hagrid_model.transform.min_size = (img_size,)
        hagrid_model.transform.max_size = img_size

        hagrid_model.transform.image_mean = img_mean
        hagrid_model.transform.image_std = img_std

        self.hagrid_model = hagrid_model

    def __call__(self, img, targets=None) -> dict:
        """
        Parameters
        ----------
        img: Tensor
            Input image
        targets: Dict
            Dictionary containing the labels for the images

        Returns
        -------
        Dict
            Dictionary containing the model output
        """
        if targets is None:
            return self.hagrid_model(img)
        else:
            model_output = self.hagrid_model(img, targets)
            return self.criterion(model_output)

    @staticmethod
    def criterion(model_output):
        """
        Parameters
        ----------
        model_output: Dict
            Dictionary containing the model output

        Returns
        -------
        Tensor
            The loss value
        """
        loss_value = sum(loss for loss in model_output.values())
        return loss_value
