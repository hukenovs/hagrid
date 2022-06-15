import os
import cv2
import time
import torch
import logging
import argparse
import numpy as np

from torch import Tensor
from PIL import Image, ImageOps
from typing import Optional, Tuple
from torchvision.transforms import functional as f

from detector.ssd_mobilenetv3 import SSDMobilenet
from detector.model import TorchVisionModel


logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

targets = {
    1: "call",
    2: "dislike",
    3: "fist",
    4: "four",
    5: "like",
    6: "mute",
    7: "ok",
    8: "one",
    9: "palm",
    10: "peace",
    11: "rock",
    12: "stop",
    13: "stop inverted",
    14: "three",
    15: "two up",
    16: "two up inverted",
    17: "three2",
    18: "peace inverted",
    19: "no gesture"
}


class Demo:

    @staticmethod
    def preprocess(img: np.ndarray) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img)
        width, height = image.size

        image = ImageOps.pad(image, (max(width, height), max(width, height)))
        padded_width, padded_height = image.size
        image = image.resize((320, 320))

        img_tensor = f.pil_to_tensor(image)
        img_tensor = f.convert_image_dtype(img_tensor)
        img_tensor = img_tensor[None, :, :, :]
        return img_tensor, (width, height), (padded_width, padded_height)

    @staticmethod
    def run(detector: TorchVisionModel, num_hands: int = 2, threshold: float = 0.5) -> None:
        """
        Run detection model and draw bounding boxes on frame
        Parameters
        ----------
        detector : TorchVisionModel
            Detection model
        num_hands:
            Min hands to detect
        threshold : float
            Confidence threshold
        """

        cap = cv2.VideoCapture(0)

        t1 = cnt = 0
        while cap.isOpened():
            delta = (time.time() - t1)
            t1 = time.time()

            ret, frame = cap.read()
            if ret:
                processed_frame, size, padded_size = Demo.preprocess(frame)
                with torch.no_grad():
                    output = detector(processed_frame)[0]
                boxes = output["boxes"][:num_hands]
                scores = output["scores"][:num_hands]
                labels = output["labels"][:num_hands]
                for i in range(min(num_hands, len(boxes))):
                    if scores[i] > threshold:

                        width, height = size
                        padded_width, padded_height = padded_size
                        scale = max(width, height) / 320

                        padding_w = abs(padded_width - width) // (2 * scale)
                        padding_h = abs(padded_height - height) // (2 * scale)

                        x1 = int((boxes[i][0] - padding_w) * scale)
                        y1 = int((boxes[i][1] - padding_h) * scale)
                        x2 = int((boxes[i][2] - padding_w) * scale)
                        y2 = int((boxes[i][3] - padding_h) * scale)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, thickness=3)
                        cv2.putText(frame, targets[int(labels[i])], (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)

                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {cnt}", (30, 30), FONT, 1, COLOR, 2)
                cnt += 1

                cv2.imshow('Frame', frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    return
            else:
                cap.release()
                cv2.destroyAllWindows()


def _load_model(model_path: str, device: str) -> TorchVisionModel:
    """
    Load model
    Parameters
    ----------
    model_path: str
        Model Path
    device: str
        Device cpu or cuda
    """
    ssd_mobilenet = SSDMobilenet(num_classes=len(targets) + 1)
    if not os.path.exists(model_path):
        logging.info(f"Model not found {model_path}")
        raise FileNotFoundError

    ssd_mobilenet.load_state_dict(model_path, map_location=device)
    ssd_mobilenet.eval()
    return ssd_mobilenet


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classifier...")

    parser.add_argument(
        "-p",
        "--path_to_model",
        required=True,
        type=str,
        help="Path to model"
    )

    parser.add_argument(
        "-d",
        "--device",
        required=False,
        default="cpu",
        type=str,
        help="Device"
    )

    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == '__main__':
    args = parse_arguments()
    model = _load_model(os.path.expanduser(args.path_to_model), args.device)
    if model is not None:
        Demo.run(model, num_hands=100, threshold=0.5)
