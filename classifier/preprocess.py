import numpy as np

from PIL import Image
from torch import nn, Tensor
from typing import Tuple, Dict, Optional, List
from torchvision.transforms import functional as F


def get_crop_from_bbox(image: Image.Image, bbox: List, box_scale: float = 1.) -> Tuple[Image.Image, np.array]:
    """
    Crop bounding box from image

    Parameters
    ----------
    image : Image.Image
        Source image for crop
    bbox : List
        Bounding box [xyxy]
    box_scale: float
        Scale for bounding box crop
    """
    int_bbox = np.array(bbox).round().astype(np.int32)

    x1 = int_bbox[0]
    y1 = int_bbox[1]
    x2 = int_bbox[2]
    y2 = int_bbox[3]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    w = h = max(x2 - x1, y2 - y1)
    x1 = max(0, cx - box_scale * w // 2)
    y1 = max(0, cy - box_scale * h // 2)
    x2 = cx + box_scale * w // 2
    y2 = cy + box_scale * h // 2
    x1, y1, x2, y2 = list(map(int, (x1, y1, x2, y2)))

    crop_image = image.crop((x1, y1, x2, y2))
    bbox_orig = np.array([x1, y1, x2, y2]).reshape(2, 2)

    return crop_image, bbox_orig


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
            image: Tensor,
            target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target


def get_transform() -> Compose:
    transforms = [ToTensor()]
    return Compose(transforms)
