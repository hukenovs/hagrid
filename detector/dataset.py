import json
import logging
import os
import random
from typing import List, Tuple

import pandas as pd
import torch.utils.data
from omegaconf import DictConfig
from PIL import Image, ImageOps

from constants import IMAGES
from detector.preprocess import Compose


class GestureDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for gesture detection pipeline
    """

    def __init__(self, is_train: bool, conf: DictConfig, transform: Compose = None, is_test: bool = False) -> None:

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

        self.labels = {
            label: num + 1 for (label, num) in zip(self.conf.dataset.targets, range(len(self.conf.dataset.targets)))
        }

        self.leading_hand = {"right": 0, "left": 1}

        subset = self.conf.dataset.get("subset", None)

        self.annotations = self.__read_annotations(subset)

        users = self.annotations["user_id"].unique()
        users = sorted(users)
        random.Random(self.conf.random_state).shuffle(users)

        train_users = users[: int(len(users) * 0.8)]
        val_users = users[int(len(users) * 0.8) :]

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
                json_annotation = json.load(open(os.path.join(path_to_json, f"{target}.json")))

                json_annotation = [
                    dict(annotation, **{"name": f"{name}.jpg"})
                    for name, annotation in zip(json_annotation, json_annotation.values())
                ]

                annotation = pd.DataFrame(json_annotation)

                annotation["target"] = target
                annotations_all = pd.concat([annotations_all, annotation], ignore_index=True)
                exists_images.extend(
                    self.__get_files_from_dir(os.path.join(self.conf.dataset.dataset, target), IMAGES, subset)
                )
            else:
                logging.info(f"Database for {target} not found")

        annotations_all["exists"] = annotations_all["name"].isin(exists_images)

        return annotations_all[annotations_all["exists"]]

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, index: int):
        row = self.annotations.iloc[[index]].to_dict("records")[0]

        image_pth = os.path.join(self.conf.dataset.dataset, row["target"], row["name"])

        image = Image.open(image_pth).convert("RGB")

        labels = torch.LongTensor([self.labels[label] for label in row["labels"]])

        target = {}
        width, height = image.size

        bboxes = []

        for bbox in row["bboxes"]:
            x1, y1, w, h = bbox
            bbox_abs = [x1 * width, y1 * height, (x1 + w) * width, (y1 + h) * height]
            bboxes.append(bbox_abs)

        bboxes_tensor = torch.as_tensor(bboxes, dtype=torch.float32)

        area = (bboxes_tensor[:, 3] - bboxes_tensor[:, 1]) * (bboxes_tensor[:, 2] - bboxes_tensor[:, 0])

        iscrowd = torch.zeros((len(row["bboxes"]),), dtype=torch.int64)
        image_id = torch.tensor([index])

        target["labels"] = labels
        target["boxes"] = bboxes_tensor
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd
        target["area"] = area

        orig_width, orig_height = image.size
        image = ImageOps.pad(image, (max(image.size), max(image.size)))
        padded_width, padded_height = image.size
        padding_w = abs(padded_width - orig_width) // 2
        padding_h = abs(padded_height - orig_height) // 2

        image = image.resize(self.conf.dataset.image_size)

        resized_boxes = []
        for bbox in target["boxes"]:
            resized_box = []
            resized_box.append((bbox[0] + padding_w) / (padded_width / self.conf.dataset.image_size[0]))
            resized_box.append((bbox[1] + padding_h) / (padded_height / self.conf.dataset.image_size[1]))
            resized_box.append((bbox[2] + padding_w) / (padded_width / self.conf.dataset.image_size[0]))
            resized_box.append((bbox[3] + padding_h) / (padded_height / self.conf.dataset.image_size[1]))

            resized_boxes.append(resized_box)

        target["boxes"] = torch.tensor(resized_boxes)

        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target
