import argparse
import logging
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.transforms import functional as f

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

from constants import targets
from detector.models.model import TorchVisionModel
from detector.utils import build_model

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


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
    def run(detector: TorchVisionModel, num_hands: int = 2, threshold: float = 0.5, landmarks: bool = False) -> None:
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
        landmarks : bool
            Detect landmarks
        """

        if landmarks:
            hands = mp.solutions.hands.Hands(
                model_complexity=0, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8
            )

        cap = cv2.VideoCapture(0)

        t1 = cnt = 0
        while cap.isOpened():
            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if ret:
                processed_frame, size, padded_size = Demo.preprocess(frame)
                with torch.no_grad():
                    output = detector(processed_frame)[0]
                boxes = output["boxes"][:num_hands]
                scores = output["scores"][:num_hands]
                labels = output["labels"][:num_hands]
                if landmarks:
                    results = hands.process(frame[:, :, ::-1])
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                mp_drawing_styles.DrawingSpec(color=[0, 255, 0], thickness=2, circle_radius=1),
                                mp_drawing_styles.DrawingSpec(color=[255, 255, 255], thickness=1, circle_radius=1),
                            )
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
                        cv2.putText(
                            frame,
                            targets[int(labels[i])],
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (0, 0, 255),
                            thickness=3,
                        )
                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {cnt}", (30, 30), FONT, 1, COLOR, 2)
                cnt += 1

                cv2.imshow("Frame", frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    return
            else:
                cap.release()
                cv2.destroyAllWindows()


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo detection...")

    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")

    parser.add_argument("-lm", "--landmarks", required=False, action="store_true", help="Use landmarks")

    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == "__main__":
    args = parse_arguments()
    conf = OmegaConf.load(args.path_to_config)
    model = build_model(
        model_name=conf.model.name,
        num_classes=len(conf.dataset.targets) + 1,
        checkpoint=conf.model.get("checkpoint", None),
        device=conf.device,
        pretrained=conf.model.pretrained,
    )

    model.eval()
    if model is not None:
        Demo.run(model, num_hands=100, threshold=0.8, landmarks=args.landmarks)
