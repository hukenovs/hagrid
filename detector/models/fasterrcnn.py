import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detector.models.model import TorchVisionModel


class FasterRCNN_Mobilenet_large(TorchVisionModel):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        torchvision_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=pretrained, pretrained_backbone=pretrained, num_classes=num_classes
        )
        self.num_classes = num_classes
        in_features = torchvision_model.roi_heads.box_predictor.cls_score.in_features
        torchvision_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        self.torchvision_model = torchvision_model

    def __call__(self, img, targets=None):
        if targets is None:
            return self.torchvision_model(img)
        else:
            return self.torchvision_model(img, targets)

    @staticmethod
    def criterion(model_output, target=None):
        loss_value = sum(loss for loss in model_output.values())
        print(f"loss_value: {loss_value}")
        return loss_value

    def to(self, device: str):
        self.torchvision_model.to(device)

    def parameters(self):
        return self.torchvision_model.parameters()

    def train(self):
        return self.torchvision_model.train()

    def eval(self):
        return self.torchvision_model.eval()

    def load_state_dict(self, checkpoint):
        self.torchvision_model.load_state_dict(checkpoint)

    def state_dict(self):
        return self.torchvision_model.state_dict()
