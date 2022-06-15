from torch import nn, Tensor
from collections.abc import Callable
from typing import Tuple, Dict, List, Iterator


class TorchVisionModel:
    """
    Torchvision class wrapper
    """
    @staticmethod
    def criterion(model_output: Dict):
        raise NotImplementedError

    def get_criterion(self) -> Callable:
        return self.criterion

    def __call__(self, img: Tensor, targets: Dict = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        raise NotImplementedError

    def to(self, device: str):
        raise NotImplementedError

    def parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError

    def train(self) -> nn.Module:
        raise NotImplementedError

    def eval(self) -> nn.Module:
        raise NotImplementedError

    def load_state_dict(self, checkpoint_path: str, map_location: str = None):
        raise NotImplementedError
