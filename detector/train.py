import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from detector.utils import add_metrics_to_tensorboard, add_params_to_tensorboard, collate_fn, save_checkpoint

try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP

    MeanAveragePrecision = MAP

logger = logging.getLogger(__name__)


class TrainDetector:
    @staticmethod
    def eval(
        model: nn.Module,
        conf: DictConfig,
        epoch: int,
        test_loader: torch.utils.data.DataLoader,
        writer: SummaryWriter,
        mode: str = "valid",
    ) -> float:
        model.eval()
        mapmetric = MeanAveragePrecision()
        if test_loader is not None:
            with torch.no_grad():
                with tqdm(test_loader, unit="batch") as tepoch:
                    tepoch.set_description(f"Eval Epoch {epoch}")
                    for i, (images, targets) in enumerate(tepoch):
                        images = list(image.to(conf.device) for image in images)
                        output = model(images)
                        for pred_box, true_box in zip(output, targets):
                            for key in true_box:
                                true_box[key] = true_box[key].to(conf.device)
                            mapmetric.update([pred_box], [true_box])

                logging.info("Start compute metric")
                mAP = mapmetric.compute()

        add_metrics_to_tensorboard(writer, mAP, epoch, mode, target="gesture")

        return mAP["map"].item()

    @staticmethod
    def run_epoch(
        model: nn.Module,
        epoch: int,
        device: str,
        optimizer: torch.optim.Optimizer,
        lr_scheduler_warmup: torch.optim.lr_scheduler.LinearLR,
        train_loader: torch.utils.data.DataLoader,
        writer: SummaryWriter,
    ) -> None:
        model.train()

        if writer is not None:
            lr_scheduler_params = lr_scheduler_warmup.state_dict()
            optimizer_params = optimizer.param_groups[0]
            add_params_to_tensorboard(writer, optimizer_params, epoch, "optimizer", {"params"})
            not_logging = lr_scheduler_params.keys() - {"start_factor", "end_factor"}
            add_params_to_tensorboard(writer, lr_scheduler_params, epoch, "lr_scheduler", not_logging)
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Train Epoch {epoch}")
            for i, (images, targets) in enumerate(tepoch):

                step = i + len(train_loader) * epoch

                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                loss = sum(loss for loss in loss_dict.values())
                loss_value = loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if lr_scheduler_warmup is not None:
                    lr_scheduler_warmup.step()

                if writer is not None:
                    writer.add_scalar("loss/train", loss_value, step)

                tepoch.set_postfix(loss=loss_value, batch=i)

    @staticmethod
    def train(
        model: nn.Module,
        conf: DictConfig,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
    ) -> None:
        """
        Initialization and running training pipeline

        Parameters
        ----------
        model : nn.Module
            Model for eval
        conf : DictConfig
            Config with training params
        train_dataset : torch.utils.data.Dataset
            Custom train gesture classifxication dataset
        test_dataset : torch.utils.data.Dataset
            Custom test gesture classification dataset
        """

        experimnt_pth = f"experiments/{conf.experiment_name}"
        writer = SummaryWriter(log_dir=f"{experimnt_pth}/logs")
        writer.add_text("model/name", conf.model.name)

        epochs = conf.train_params.epochs

        model = model.to(conf.device)

        params = [p for p in model.parameters() if p.requires_grad]

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=conf.train_params.train_batch_size,
            num_workers=conf.train_params.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=conf.train_params.prefetch_factor,
            shuffle=True,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=conf.train_params.test_batch_size,
            num_workers=conf.train_params.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
            prefetch_factor=conf.train_params.prefetch_factor,
        )

        optimizer = torch.optim.SGD(
            params, lr=conf.optimizer.lr, momentum=conf.optimizer.momentum, weight_decay=conf.optimizer.weight_decay
        )

        warmup_iters = min(1000, len(train_dataloader) - 1)
        lr_scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=conf.scheduler.start_factor, total_iters=warmup_iters
        )

        best_metric = -1.0
        conf_dictionary = OmegaConf.to_container(conf)
        for epoch in range(conf.model.start_epoch, epochs):
            logging.info(f"Epoch: {epoch}")
            TrainDetector.run_epoch(model, epoch, conf.device, optimizer, lr_scheduler_warmup, train_dataloader, writer)
            current_metric_value = TrainDetector.eval(model, conf, epoch, test_dataloader, writer)
            save_checkpoint(experimnt_pth, conf_dictionary, model, optimizer, epoch, f"model_{epoch}.pth")

            if current_metric_value > best_metric:
                logging.info(f"Save best model with metric: {current_metric_value}")
                save_checkpoint(experimnt_pth, conf_dictionary, model, optimizer, epoch, "best_model.pth")
                best_metric = current_metric_value

        writer.flush()
        writer.close()
