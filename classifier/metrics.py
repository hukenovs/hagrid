from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import accuracy, auroc, confusion_matrix, f1_score, precision, recall


def get_metrics(
    targets: Tensor,
    predicts: Tensor,
    conf: DictConfig,
    epoch: int,
    mode: str,
    writer: SummaryWriter,
    target: str = "gesture",
) -> Dict:
    """
    Calc metrics for predicted labels

    Parameters
    ----------
    targets : Tensor
        Target class labels
    predicts : Tensor
        Predicted class labels
    conf : DictConfig
        Config
    epoch : int
        Number of epoch
    mode : str
        Mode valid or train
    writer : SummaryWriter
        Tensorboard log writer
    target : str
        Target name: gesture or leading_hand
    """
    average = conf.metric_params["average"]
    metrics = conf.metric_params["metrics"]
    num_classes = conf.num_classes[target]
    predicts_labels = torch.argmax(predicts, dim=1)
    scores = {
        "accuracy": accuracy(predicts_labels, targets, average=average, num_classes=num_classes),
        "f1_score": f1_score(predicts_labels, targets, average=average, num_classes=num_classes),
        "precision": precision(predicts_labels, targets, average=average, num_classes=num_classes),
        "recall": recall(predicts_labels, targets, average=average, num_classes=num_classes),
    }

    if mode == "test":
        scores["roc_auc"] = auroc(predicts, targets, average=average, num_classes=num_classes)

    needed_scores = {}
    for metric in metrics:
        needed_scores[metric] = round(float(scores[metric]), 6)

    if mode == "valid" or mode == "test":
        if target == "leading_hand":
            class_names = ["right", "left"]
        else:
            class_names = conf.dataset.targets

        cm = confusion_matrix(predicts, targets, num_classes)
        df_cm = pd.DataFrame(cm, index=[i for i in class_names], columns=[i for i in class_names])

        plt.figure(figsize=(16, 12))
        hm = sns.heatmap(df_cm, annot=True, fmt=".5g", cmap="YlGnBu").get_figure()
        writer.add_figure(f"Confusion matrix for {target}", hm, epoch)
    return needed_scores
