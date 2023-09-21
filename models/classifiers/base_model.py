from typing import Dict

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from models.model import HaGRIDModel


class ClassifierModel(HaGRIDModel):
    def __init__(self, model: nn.Module, **kwargs):
        """
        Parameters
        ----------
        model: nn.Module
            The model to be used for classification
        """
        super().__init__()
        self.hagrid_model = model(**kwargs)
        self.criterion = None

    def __call__(self, images: list[Tensor], targets: Dict = None) -> Dict:
        """
        Parameters
        ----------
        images: list[Tensor]
            List of images to be passed to the model
        targets: Dict
            Dictionary containing the labels for the images

        Returns
        -------
        Dict
            Dictionary containing the model output
        """
        image_tensors = torch.stack(images)
        model_output = self.hagrid_model(image_tensors)
        model_output = {"labels": model_output}
        if targets is None:
            return model_output
        else:
            target_tensors = torch.stack([target["labels"] for target in targets])
            return self.criterion(model_output["labels"], target_tensors)

    def criterion(self, model_output: Dict, targets: DictConfig):
        """
        Parameters
        ----------
        model_output: Dict
            Dictionary containing the model output
        targets: DictConfig
            Dictionary containing the labels for the images

        Returns
        -------
        Tensor
            The loss value
        """
        loss_value = self.criterion(model_output, targets)
        return loss_value
