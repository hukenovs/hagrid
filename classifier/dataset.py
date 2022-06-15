import os
import json
import random
import numpy as np
import pandas as pd
import logging
import torch.utils.data

from PIL import Image, ImageOps
from typing import Dict, Tuple, List
from omegaconf import DictConfig
from classifier.preprocess import get_crop_from_bbox, Compose

FORMATS = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")


class GestureDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for gesture classification pipeline
    """

    def __init__(
            self,
            is_train: bool,
            conf: DictConfig,
            transform: Compose = None,
            is_test: bool = False
    ) -> None:

        """
        Custom Dataset for gesture classification pipeline

        Parameters
        ----------
        is_train : bool
            True if collect train dataset else False
        is_test: Bool
            For metrics calculation on test set
        conf : DictConfig
            Config with training params
        transform : Compose
            Compose of transforms
        """
        self.conf = conf
        self.transform = transform
        self.is_train = is_train

        self.labels = {label: num for (label, num) in
                       zip(self.conf.dataset.targets, range(len(self.conf.dataset.targets)))}

        self.leading_hand = {"right": 0, "left": 1}

        subset = self.conf.dataset.get("subset", None)

        self.annotations = self.__read_annotations(subset)

        users = self.annotations["user_id"].unique()
        users = sorted(users)
        random.Random(self.conf.random_state).shuffle(users)

        train_users = users[:int(len(users) * 0.8)]
        val_users = users[int(len(users) * 0.8):]

        self.annotations = self.annotations.copy()

        if not is_test:
            if is_train:
                self.annotations = self.annotations[self.annotations["user_id"].isin(train_users)]
            else:
                self.annotations = self.annotations[self.annotations["user_id"].isin(val_users)]

    @staticmethod
    def __get_files_from_dir(pth: str, extns: Tuple, subset: int = None) -> List:
        """
        Get list of files from dir according to extensions(extns)

        Parameters
        ----------
        pth : str
            Path ot dir
        extns: Tuple
            Set of file extensions
        subset : int
            Length of subset for each target
        """
        if not os.path.exists(pth):
            logging.warning(f"Dataset directory doesn't exist {pth}")
            return []
        files = [f for f in os.listdir(pth) if f.endswith(extns)]
        if subset is not None:
            files = files[:subset]
        return files

    def __read_annotations(self, subset: int = None) -> pd.DataFrame:
        """
        Read annotations json

        Parameters
        ----------
        subset : int
            Length of subset for each target
        """
        exists_images = []
        annotations_all = pd.DataFrame()
        path_to_json = os.path.expanduser(self.conf.dataset.annotations)
        for target in self.conf.dataset.targets:
            target_tsv = os.path.join(path_to_json, f"{target}.json")
            if os.path.exists(target_tsv):
                json_annotation = json.load(open(
                    os.path.join(path_to_json, f"{target}.json")
                ))

                json_annotation = [dict(annotation, **{"name": f"{name}.jpg"}) for name, annotation in zip(
                    json_annotation, json_annotation.values()
                )]

                annotation = pd.DataFrame(json_annotation)

                annotation["target"] = target
                annotations_all = pd.concat([annotations_all, annotation], ignore_index=True)
                exists_images.extend(
                    self.__get_files_from_dir(os.path.join(self.conf.dataset.dataset, target),
                                              FORMATS, subset))
            else:
                logging.info(f"Databse for {target} not found")

        annotations_all["exists"] = annotations_all["name"].isin(exists_images)

        return annotations_all[annotations_all["exists"]]

    def __prepare_image_target(
            self,
            target: str,
            name: str,
            bboxes: List,
            labels: List,
            leading_hand: str
    ) -> Tuple[Image.Image, str, str]:
        """
        Crop and padding image, prepare target

        Parameters
        ----------
        target : str
            Class name
        name : str
            Name of image
        bboxes : List
            List of bounding boxes [xywh]
        labels: List
            List of labels
        leading_hand : str
            Leading hand class name
        """
        image_pth = os.path.join(self.conf.dataset.dataset, target, name)

        image = Image.open(image_pth).convert("RGB")

        width, height = image.size

        choice = np.random.choice(["gesture", "no_gesture"], p=[0.7, 0.3])

        bboxes_by_class = {}

        for i, bbox in enumerate(bboxes):
            x1, y1, w, h = bbox
            bbox_abs = [x1 * width, y1 * height, (x1 + w) * width, (y1 + h) * height]
            if labels[i] == "no_gesture":
                bboxes_by_class["no_gesture"] = (bbox_abs, labels[i])
            else:
                bboxes_by_class["gesture"] = (bbox_abs, labels[i])

        if choice not in bboxes_by_class:
            choice = list(bboxes_by_class.keys())[0]

        if self.is_train:
            box_scale = np.random.uniform(low=1.0, high=2.0)
        else:
            box_scale = 1.0

        image_cropped, bbox_orig = get_crop_from_bbox(image, bboxes_by_class[choice][0], box_scale=box_scale)

        image_resized = ImageOps.pad(image_cropped, tuple(self.conf.dataset.image_size), color=(0, 0, 0))

        gesture = bboxes_by_class[choice][1]

        leading_hand_class = leading_hand
        if gesture == "no_gesture":
            leading_hand_class = "right" if leading_hand == "left" else "left"

        return image_resized, gesture, leading_hand_class

    def __len__(self) -> int:
        return self.annotations.shape[0]

    def __getitem__(self, index: int) -> Tuple[Image.Image, Dict]:
        """
        Get item from annotations

        Parameters
        ----------
        index : int
            Index of annotation item
        """
        row = self.annotations.iloc[[index]].to_dict('records')[0]

        image_resized, gesture, leading_hand = self.__prepare_image_target(
            row["target"],
            row["name"],
            row["bboxes"],
            row["labels"],
            row["leading_hand"]
        )

        label = {"gesture": self.labels[gesture],
                 "leading_hand": self.leading_hand[leading_hand]}

        if self.transform is not None:
            image_resized, label = self.transform(image_resized, label)

        return image_resized, label
