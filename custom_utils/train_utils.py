import os
from typing import List, Tuple

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter

from custom_utils.ddp_utils import get_sampler
from custom_utils.utils import Logger, build_model, get_transform
from models import HaGRIDModel


def collate_fn(batch: List) -> Tuple:
    """
    Collate func for dataloader

    Parameters
    ----------
    batch : List
        Batch of data
    """
    return list(zip(*batch))


def get_dataloader(dataset: Dataset, sampler: Sampler = None, **kwargs) -> DataLoader:
    """
    Get dataloader

    Parameters
    ----------
    dataset : Dataset
        Dataset
    sampler : Sampler, optional
        Sampler, by default None
    **kwargs

    Returns
    -------
    DataLoader
        Dataloader
    """
    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        shuffle=kwargs["shuffle"] if sampler is None else False,
        sampler=sampler,
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
        prefetch_factor=kwargs["prefetch_factor"],
    )


def load_train_objects(config: DictConfig, command: str, n_gpu: int):
    """
    Load train objects

    Parameters
    ----------
    config : DictConfig
        Config
    command : str [train, test]
        Command
    n_gpu : int
        Number of gpus

    Returns
    -------
    Tuple
        Train dataloader, validation dataloader, test dataloader, model
    """
    model = build_model(config)

    if model.type == "detector":
        from dataset import DetectionDataset as GestureDataset
    elif model.type == "classifier":
        from dataset import ClassificationDataset as GestureDataset
    else:
        raise Exception(f"Model type {model.type} does not exist")

    test_dataset = GestureDataset(config, "test", get_transform(config.test_transforms, model.type))

    if command == "train":
        train_dataset = GestureDataset(config, "train", get_transform(config.train_transforms, model.type))
        if config.dataset.dataset_val and config.dataset.annotations_val:
            val_dataset = GestureDataset(config, "val", get_transform(config.val_transforms, model.type))
        else:
            raise Exception("Cannot train without validation data")

    train_sampler = None
    test_sampler = None
    val_sampler = None
    if n_gpu > 1:
        test_sampler = get_sampler(test_dataset)
        if command == "train":
            train_sampler = get_sampler(train_dataset)
            if val_dataset:
                val_sampler = get_sampler(val_dataset)

    test_dataloader = get_dataloader(test_dataset, test_sampler, **config.test_params)
    if command == "train":
        train_dataloader = get_dataloader(train_dataset, train_sampler, **config.train_params)
        if val_dataset:
            val_dataloader = get_dataloader(val_dataset, val_sampler, **config.val_params)
    else:
        train_dataloader = None
        val_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader, model


def load_train_optimizer(model: HaGRIDModel, config: DictConfig):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = getattr(torch.optim, config.optimizer.name)(parameters, **config.optimizer.params)
    if config.scheduler.name:
        scheduler = getattr(torch.optim.lr_scheduler, config.scheduler.name)(optimizer, **config.scheduler.params)
    else:
        scheduler = None
    return optimizer, scheduler


class Trainer:
    def __init__(
        self,
        model: HaGRIDModel,
        config: DictConfig,
        test_data: torch.utils.data.DataLoader,
        train_data: torch.utils.data.DataLoader = None,
        val_data: torch.utils.data.DataLoader = None,
        metric_calculator=None,
        n_gpu: int = 1,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
    ) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.gpu_id = int(os.environ["LOCAL_RANK"]) if n_gpu > 1 else 0
        self.model = model
        self.model.to(f"cuda:{self.gpu_id}")

        if self.model.type == "classifier":
            self.metric_name = "F1Score"
        else:
            self.metric_name = "map"
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.current_state = {
            "loss": 0,
            "metric": {self.metric_name: 0},
            "epoch": 0,
        }
        self.best_state = {
            "loss": 0,
            "metric": {self.metric_name: 0},
            "epoch": 0,
        }

        self.stop = False
        self.max_epoch = self.config.epochs
        self.epochs_run = 0
        self.n_gpu = n_gpu
        self.metric_calculator = metric_calculator.to(f"cuda:{self.gpu_id}")
        if self.n_gpu > 1:
            self.model.hagrid_model = SyncBatchNorm.convert_sync_batchnorm(self.model.hagrid_model)
            self.model.hagrid_model = DDP(self.model.hagrid_model, device_ids=[self.gpu_id])

        if self.gpu_id == 0:
            if not os.path.exists(self.config.work_dir):
                os.mkdir(self.config.work_dir)

            self.summary_writer = SummaryWriter(log_dir=f"{self.config.work_dir}/{self.config.experiment_name}/logs")
            self.summary_writer.add_text("model/name", self.config.model.name)

        if self.config.model.checkpoint is not None:
            self._load_snapshot(self.config.model.checkpoint)

    def _save_snapshot(self):
        metric_score = self.best_state["metric"][self.metric_name]
        if self.n_gpu > 1:
            state = self.model.hagrid_model.module.state_dict()
        else:
            state = self.model.state_dict()
        snapshot = {
            "MODEL_STATE": state,
            "OPTIMIZER_STATE": self.optimizer.state_dict(),
            "SCHEDULER_STATE": self.scheduler.state_dict() if self.scheduler else None,
            "EPOCHS_RUN": self.best_state["epoch"],
            "Loss": self.best_state["loss"],
            "Metric": self.best_state["metric"],
        }
        save_path = os.path.join(self.config.work_dir, self.config.experiment_name)
        save_name = f"{self.config.model.name}_epoch:{self.best_state['epoch']}_{self.metric_name}:{metric_score:.2}_loss:{self.best_state['loss']:.2}.pth"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(snapshot, os.path.join(save_path, save_name))
        print(f"Save model {self.config.model.name} || {self.metric_name}:{metric_score:.2}")

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        if self.scheduler:
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.best_state["epoch"] = snapshot["EPOCHS_RUN"]
        self.best_state["loss"] = snapshot["Loss"]
        self.best_state["metric"] = snapshot["Metric"]
        print(f"Loaded model from {snapshot_path}")

    def test(self):
        self.model.eval()
        if self.test_data is None:
            raise Exception("Cannot test without test data")

        with Logger("Test", self.max_epoch, len(self.test_data), self.config.log_every, self.gpu_id) as logger:
            for iteration, (images, targets) in enumerate(self.test_data):
                images = list(image.to(self.gpu_id) for image in images)
                targets = [{k: v.to(self.gpu_id) for k, v in t.items()} for t in targets]

                with torch.no_grad():
                    output = self.model(images)

                # TODO: use it for fix CUDA OOM
                # targets = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
                # output = [{k: v.detach().cpu() for k, v in t.items()} for t in output]

                metric = self.metric_calculator(output, targets)

                logger.log_iteration(iteration + 1, self.current_state["epoch"], metrics=metric)

            if self.gpu_id == 0:
                for key, value in metric.items():
                    self.summary_writer.add_scalar(f"{key}/Test", value, self.current_state["epoch"])

    def val(self):
        self.model.eval()
        if self.val_data is None:
            raise Exception("Cannot validate without validation data")
        with Logger("Eval", self.max_epoch, len(self.val_data), self.config.log_every, self.gpu_id) as logger:
            for iteration, (images, targets) in enumerate(self.val_data):
                images = list(image.to(self.gpu_id) for image in images)
                targets = [{k: v.to(self.gpu_id) for k, v in t.items()} for t in targets]

                with torch.no_grad():
                    output = self.model(images)

                # TODO: use it for fix CUDA OOM
                # targets = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
                # output = [{k: v.detach().cpu() for k, v in t.items()} for t in output]

                metric = self.metric_calculator(output, targets)
                logger.log_iteration(iteration + 1, self.current_state["epoch"], metrics=metric)

            if self.gpu_id == 0:
                self.current_state["metric"] = logger.metric_averager.value

                for key, value in self.current_state["metric"].items():
                    self.summary_writer.add_scalar(f"{key}/Eval", value, self.current_state["epoch"])

                if (
                    self.current_state["metric"][self.metric_name] - self.best_state["metric"][self.metric_name]
                ) > self.config.early_stopping.metric:
                    self.best_state["metric"] = self.current_state["metric"]
                    self.best_state["loss"] = self.current_state["loss"]
                    self.best_state["epoch"] = self.current_state["epoch"]

                    self._save_snapshot()

    def early_stop(self):
        if (
            self.current_state["epoch"] - self.best_state["epoch"] >= self.config.early_stopping.epochs
            and self.current_state["metric"][self.metric_name] - self.best_state["metric"][self.metric_name]
            <= self.config.early_stopping.metric
        ):
            return True
        else:
            return False

    def train(self):
        if self.train_data is None:
            raise Exception("Cannot train without training data")
        for epoch in range(self.epochs_run, self.max_epoch):
            if self.gpu_id == 0:
                if self.early_stop():
                    self.stop = True
                    if self.n_gpu > 1:
                        torch.distributed.broadcast(self.stop, 0)
            if self.stop:
                break

            self.model.train()
            self.current_state["epoch"] = epoch
            if self.n_gpu > 1:
                self.train_data.sampler.set_epoch(self.current_state["epoch"])
            with Logger("Train", self.max_epoch, len(self.train_data), self.config.log_every, self.gpu_id) as logger:
                for iteration, (images, targets) in enumerate(self.train_data):
                    self.optimizer.zero_grad()

                    images = list(image.to(self.gpu_id) for image in images)
                    targets = [{k: v.to(self.gpu_id) for k, v in t.items()} for t in targets]

                    loss = self.model(images, targets)

                    loss.backward()
                    self.optimizer.step()

                    if self.n_gpu > 1:
                        world_size = dist.get_world_size()
                        dist.reduce(loss, dst=0, op=dist.ReduceOp.SUM)
                        loss = loss / world_size

                    logger.log_iteration(iteration + 1, self.current_state["epoch"], loss.item())

                if self.scheduler is not None:
                    if self.config.scheduler.name == "ReduceLROnPlateau":
                        self.scheduler.step(self.current_state["loss"])
                    else:
                        self.scheduler.step()

                if self.gpu_id == 0:
                    self.current_state["loss"] = logger.loss_averager.value
                    self.summary_writer.add_scalar(
                        "loss/Train", self.current_state["loss"], self.current_state["epoch"]
                    )

            if self.config.eval_every > 0 and self.current_state["epoch"] % self.config.eval_every == 0:
                self.val()

            if self.config.test_every > 0 and self.current_state["epoch"] % self.config.test_every == 0:
                self.test()
