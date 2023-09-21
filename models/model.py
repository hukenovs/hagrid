from typing import Dict, Iterator, List, Tuple

from torch import Tensor, nn


class HaGRIDModel:
    """
    Torchvision class wrapper
    """

    def __init__(self):
        self.hagrid_model = None
        self.type = None

    def __call__(self, img: Tensor, targets: Dict = None) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
        raise NotImplementedError

    def to(self, device: str):
        self.hagrid_model = self.hagrid_model.to(device)

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.hagrid_model.parameters()

    def train(self):
        self.hagrid_model.train()

    def eval(self):
        self.hagrid_model.eval()

    def load_state_dict(self, state_dict: Dict[str, Tensor]):
        self.hagrid_model.load_state_dict(state_dict)

    def state_dict(self):
        return self.hagrid_model.state_dict()
