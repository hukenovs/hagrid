import argparse
from typing import Optional, Tuple

from omegaconf import OmegaConf
from torch.distributed import destroy_process_group

try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP

    MeanAveragePrecision = MAP

from custom_utils.ddp_utils import ddp_setup
from custom_utils.train_utils import Trainer, load_train_objects, load_train_optimizer
from custom_utils.utils import F1ScoreWithLogging


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture classifier...")

    parser.add_argument(
        "-c", "--command", required=True, type=str, help="Training or test pipeline", choices=("train", "test")
    )

    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")

    parser.add_argument("--n_gpu", required=False, type=int, default=1, help="Number of GPUs to use")

    known_args, _ = parser.parse_known_args(params)
    return known_args


def run(args):
    config = OmegaConf.load(args.path_to_config)

    if args.n_gpu > 1:
        ddp_setup()

    train_dataloader, val_dataloader, test_dataloader, model = load_train_objects(config, args.command, args.n_gpu)

    if model.type == "detector":
        metric = MeanAveragePrecision(class_metrics=False)
    else:
        task = "binary" if config.dataset.one_class else "multiclass"
        num_classes = 2 if config.dataset.one_class else len(config.dataset.targets)
        metric = F1ScoreWithLogging(task=task, num_classes=num_classes)

    optimizer, scheduler = load_train_optimizer(model, config)
    trainer = Trainer(
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_calculator=metric,
        train_data=train_dataloader,
        val_data=val_dataloader,
        test_data=test_dataloader,
        n_gpu=args.n_gpu,
    )

    if args.command == "train":
        trainer.train()

    if args.command == "test":
        trainer.test()

    if args.n_gpu > 1:
        destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
