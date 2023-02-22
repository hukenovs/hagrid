from typing import Dict, List, Optional, Tuple

from torch import Tensor, nn
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms: List[nn.Module]):
        self.transforms = transforms

    def __call__(self, image, target) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(nn.Module):
    @staticmethod
    def forward(
        image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target


def get_transform() -> Compose:
    transforms = [ToTensor()]
    return Compose(transforms)
