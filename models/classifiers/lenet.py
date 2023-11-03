import torch
import torch.nn.functional as F


class LeNet(torch.nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        """
        Torchvision two headed MobileNet V3 configuration

        Parameters
        ----------
        num_classes : int
            Number of classes for each task
        """

        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5, 2)
        self.conv3 = torch.nn.Conv2d(16, 32, 5, 2)
        self.fc1 = torch.nn.Linear(in_features=8 * 4 * 4, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.gesture_classifier = torch.nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        gesture = self.gesture_classifier(x)
        return gesture

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
