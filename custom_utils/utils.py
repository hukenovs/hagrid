from collections import defaultdict
from time import gmtime, strftime
from typing import Dict

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torchmetrics import F1Score

from models import classifiers_list, detectors_list


class F1ScoreWithLogging:
    def __init__(self, task, num_classes):
        """
        Wrapper for F1Score metric from torchmetrics with logging

        Parameters
        ----------
        task : str
            Task for F1Score metric
        num_classes : int
            Number of classes for F1Score metric
        """
        self.f1_score = F1Score(task=task, num_classes=num_classes)

    def to(self, device):
        """
        Move metric to device

        Parameters
        ----------
        device : torch.device
            Device to move metric
        """
        self.f1_score = self.f1_score.to(device)
        return self

    def __call__(self, preds, targets):
        """
        Calculate F1Score metric

        Parameters
        ----------
        preds : dict
            Predictions from model
        targets : list
            Targets from dataset
        """
        target = torch.stack([target["labels"] for target in targets])
        result = self.f1_score(preds["labels"].argmax(1), target)
        return {"F1Score": result}


class Logger:
    def __init__(self, train_state: str, max_epochs: int, dataloader_len: int, log_every: int, gpu_id: int):
        """
        Logger for training and evaluation

        Parameters
        ----------
        train_state : str
            Train state: Train, Eval or Test
        max_epochs : int
            Number of epochs
        dataloader_len : int
            Length of dataloader
        log_every : int
            Log every n iterations
        gpu_id : int
            Id of gpu
        """
        self.dataloader_len = dataloader_len
        self.max_epochs = max_epochs
        self.train_state = train_state
        self.log_every = log_every
        self.gpu_id = gpu_id
        self.loss_averager = LossAverager()
        self.metric_averager = MetricAverager()

    def log_iteration(self, iteration: int, epoch: int, loss: float = None, metrics: dict = None):
        """
        Log iteration

        Parameters
        ----------
        iteration : int
            Current iteration
        epoch : int
            Current epoch
        loss : float, optional
            Loss value, by default None
        metrics : dict, optional
            Metrics values, by default None
        """
        if self.gpu_id != 0:
            return
        if (iteration % self.log_every == 0) or (iteration == self.dataloader_len):
            log_str = f"Time: {strftime('%Y-%m-%d %H:%M:%S', gmtime())} "
            log_str += f"{self.train_state} ---- Epoch [{epoch}/{self.max_epochs}], Iteration [{iteration}/{self.dataloader_len}]:"
            if self.train_state == "Train" and loss is not None:
                self.loss_averager.update(loss)
                log_str += f" Loss: {self.loss_averager.value:.4f}"
            if self.train_state in ["Eval", "Test"] and metrics is not None:
                try:
                    del metrics["classes"]
                except KeyError:
                    pass
                self.metric_averager.update(metrics)
                if iteration == self.dataloader_len:
                    for metric_name, metric_value in self.metric_averager.value.items():
                        log_str += f" {metric_name}: {metric_value:.3}"
            print(log_str)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class MetricAverager:
    def __init__(self):
        self.current_total = defaultdict(float)
        self.iterations = 0

    def update(self, values: Dict):
        for key, value in values.items():
            self.current_total[key] += value.item()
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            metrics = {key: value / self.iterations for key, value in self.current_total.items()}
            return metrics


class LossAverager:
    def __init__(self):
        self.iterations = 0
        self.current_total = 0

    def update(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return self.current_total / self.iterations


def get_transform(transform_config: DictConfig, model_type: str):
    transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
    transforms_list.append(ToTensorV2())
    if model_type == "detector":
        return A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0, label_fields=["class_labels"]),
        )
    elif model_type == "classifier":
        return A.Compose(transforms_list)


def build_model(config: DictConfig):
    model_name = config.model.name
    model_config = {"num_classes": len(config.dataset.targets), "pretrained": config.model.pretrained}
    if model_name in detectors_list:
        model_config["num_classes"] += 1
        model_config.update(
            {
                "pretrained_backbone": config.model.pretrained_backbone,
                "img_size": config.dataset.img_size,
                "img_mean": config.dataset.img_mean,
                "img_std": config.dataset.img_std,
            }
        )
        model = detectors_list[model_name](**model_config)
        model.type = "detector"
    elif model_name in classifiers_list:
        model = classifiers_list[model_name](**model_config)
        model.criterion = getattr(torch.nn, config.criterion)()
        model.type = "classifier"
    else:
        raise Exception(f"Unknown model {model_name}")

    return model
