import argparse
import logging
import time
from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from constants import targets
from custom_utils.utils import build_model

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Demo:
    @staticmethod
    def preprocess(img: np.ndarray, transform) -> Tuple[Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Preproc image for model input
        Parameters
        ----------
        img: np.ndarray
            input image
        transform :
            albumentation transforms
        """
        transformed_image = transform(image=img)
        return transformed_image["image"]

    @staticmethod
    def get_transform_for_inf(transform_config: DictConfig):
        """
        Create list of transforms from config
        Parameters
        ----------
        transform_config: DictConfig
            config with test transforms
        """
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)

    @staticmethod
    def run(classifier, transform) -> None:
        """
        Run detection model and draw bounding boxes on frame
        Parameters
        ----------
        classifier : TorchVisionModel
            Classifier model
        transform :
            albumentation transforms
        """

        cap = cv2.VideoCapture(0)
        t1 = cnt = 0
        while cap.isOpened():
            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if ret:
                processed_frame = Demo.preprocess(frame, transform)
                with torch.no_grad():
                    output = classifier([processed_frame])
                label = output["labels"].argmax(dim=1)

                cv2.putText(
                    frame, targets[int(label)], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3
                )
                fps = 1 / delta
                cv2.putText(frame, f"FPS: {fps :02.1f}, Frame: {cnt}", (30, 30), FONT, 1, (255, 0, 255), 2)
                cnt += 1

                cv2.imshow("Frame", frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    return
            else:
                cap.release()
                cv2.destroyAllWindows()


def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo full frame classification...")

    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")

    known_args, _ = parser.parse_known_args(params)
    return known_args


if __name__ == "__main__":
    args = parse_arguments()
    conf = OmegaConf.load(args.path_to_config)
    model = build_model(conf)
    transform = Demo.get_transform_for_inf(conf.test_transforms)
    if conf.model.checkpoint is not None:
        snapshot = torch.load(conf.model.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(snapshot["MODEL_STATE"])
    model.eval()
    if model is not None:
        Demo.run(model, transform)
