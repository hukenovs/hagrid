from typing import Dict, List, Tuple, Union

import torchvision
import torchvision.models.detection._utils as det_utils
from torch import Tensor
from torchvision.models.detection.ssd import SSDClassificationHead

from models.model import HaGRIDModel


class SSD300_Vgg16(HaGRIDModel):
    """
    Torchvision SSD model for gesture detection
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = False,
        pretrained_backbone: bool = False,
        img_size: Union[int, tuple] = 300,
        img_mean: list = [0.485, 0.456, 0.406],
        img_std: list = [0.229, 0.224, 0.225],
    ):
        """
        Torchvision SSD model for gesture detection

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

        hagrid_model = torchvision.models.detection.ssd300_vgg16(
            pretrained=pretrained, pretrained_backbone=pretrained_backbone
        )

        in_channels = det_utils.retrieve_out_channels(hagrid_model.backbone, (img_size, img_size))
        num_anchors = hagrid_model.anchor_generator.num_anchors_per_location()

        hagrid_model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        hagrid_model.transform.min_size = (img_size,)
        hagrid_model.transform.max_size = img_size

        hagrid_model.transform.image_mean = img_mean
        hagrid_model.transform.image_std = img_std

        self.hagrid_model = hagrid_model

    def __call__(
        self, img: Tensor, targets: Dict = None
    ) -> Union[Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]], float]:
        """
        Parameters
        ----------
        img: Tensor
            Input image
        targets: Dict
            Targets for model

        Returns
        -------
        Union[Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]], float]
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
        Calculate loss value

        Parameters
        ----------
        model_output: Dict
            Dictionary containing the model output

        Returns
        -------
        float
            Loss value
        """
        loss_value = sum(loss for loss in model_output.values())
        return loss_value
