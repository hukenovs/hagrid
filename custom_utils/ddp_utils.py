import os

import torch
from torch.distributed import init_process_group
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


def get_sampler(dataset: Dataset) -> DistributedSampler:
    """
    Returns a DistributedSampler for the given dataset.

    Parameters
    ----------
    dataset : Dataset
        The dataset to sample from.
    """
    return DistributedSampler(dataset)


def ddp_setup():
    """
    Sets up the distributed data parallel environment.
    """
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
