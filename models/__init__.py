from functools import partial

from torchvision import models

from .classifiers import ClassifierModel, LeNet, VitB16, VitB32
from .detectors import RetinaNet_ResNet50, SSD300_Vgg16, SSDLiteMobilenet_large, SSDLiteMobilenet_small
from .model import HaGRIDModel

detectors_list = {
    "SSDLiteMobileNetV3Large": SSDLiteMobilenet_large,
    "SSDLiteMobileNetV3Small": SSDLiteMobilenet_small,
    "SSD300_Vgg16": SSD300_Vgg16,
    "RetinaNet_ResNet50": RetinaNet_ResNet50,
}

classifiers_list = {
    "LeNet": LeNet,
    "ResNet18": partial(ClassifierModel, models.resnet18),
    "ResNet50": partial(ClassifierModel, models.resnet50),
    "ResNet152": partial(ClassifierModel, models.resnet152),
    "ResNext50": partial(ClassifierModel, models.resnext50_32x4d),
    "ResNext101": partial(ClassifierModel, models.resnext101_32x8d),
    "MobileNetV3_small": partial(ClassifierModel, models.mobilenet_v3_small),
    "MobileNetV3_large": partial(ClassifierModel, models.mobilenet_v3_large),
    "VitB32": partial(ClassifierModel, VitB32),
    "VitB16": partial(ClassifierModel, VitB16),
}
