from functools import partial

from torchvision import models

from .classifiers import ClassifierModel, VitB16
from .detectors import SSDLiteMobilenet_large
from .model import HaGRIDModel

detectors_list = {
    "SSDLiteMobileNetV3Large": SSDLiteMobilenet_large,
}

classifiers_list = {
    "ResNet18": partial(ClassifierModel, models.resnet18),
    "ResNet152": partial(ClassifierModel, models.resnet152),
    "MobileNetV3_small": partial(ClassifierModel, models.mobilenet_v3_small),
    "MobileNetV3_large": partial(ClassifierModel, models.mobilenet_v3_large),
    "VitB16": partial(ClassifierModel, VitB16),
    "ConvNeXt_base": partial(ClassifierModel, models.convnext_base),
}
