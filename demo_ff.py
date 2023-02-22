import argparse
import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import Tensor
from torchvision.transforms import functional as f

from classifier.utils import build_model
from constants import targets

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

        image = image.resize((224, 224))

        img_tensor = f.pil_to_tensor(image)
        img_tensor = f.convert_image_dtype(img_tensor)
        img_tensor = img_tensor[None, :, :, :]
        return img_tensor, (width, height)

    @staticmethod
    def run(classifier) -> None:
        """
        Run detection model and draw bounding boxes on frame
        Parameters
        ----------
        classifier : TorchVisionModel
            Classifier model
        """

        cap = cv2.VideoCapture(0)
        t1 = cnt = 0
        while cap.isOpened():
            delta = time.time() - t1
            t1 = time.time()

            ret, frame = cap.read()
            if ret:
                processed_frame, size = Demo.preprocess(frame)
                with torch.no_grad():
                    output = classifier(processed_frame)
                label = output["gesture"].argmax(dim=1)

                cv2.putText(
                    frame, targets[int(label) + 1], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3
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
    if not conf.model.full_frame:
        raise Exception("For Full Frame model use full-frame: True in config")
    model = build_model(
        model_name=conf.model.name,
        num_classes=len(targets) - 1,
        checkpoint=conf.model.get("checkpoint", None),
        device=conf.device,
        pretrained=conf.model.pretrained,
        freezed=conf.model.freezed,
        ff=conf.model.full_frame,
    )
    model.eval()
    if model is not None:
        Demo.run(model)
