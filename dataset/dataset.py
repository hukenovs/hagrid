import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import IMAGES


class HagridDataset(Dataset):
    """
    Custom Dataset for HaGRID
    """

    def __init__(self, conf: DictConfig, dataset_type: str, transform):
        """
        Parameters
        ----------
        conf : DictConfig
            Config for dataset
        dataset_type : str
            Type of dataset
        transform : albumentations.Compose
            Transformations for dataset
        """
        self.conf = conf
        self.labels = {
            label: num for (label, num) in zip(self.conf.dataset.targets, range(len(self.conf.dataset.targets)))
        }

        self.dataset_type = dataset_type

        subset = self.conf.dataset.get("subset", None) if dataset_type == "train" else -1

        self.path_to_json = os.path.expanduser(self.conf.dataset.get(f"annotations_{dataset_type}"))
        self.path_to_dataset = os.path.expanduser(self.conf.dataset.get(f"dataset_{dataset_type}"))
        self.annotations = self.__read_annotations(subset)

        self.transform = transform

    @staticmethod
    def _load_image(image_path: str):
        """
        Load image from path

        Parameters
        ----------
        image_path : str
            Path to image
        """
        image = Image.open(image_path).convert("RGB")

        return image

    @staticmethod
    def __get_files_from_dir(pth: str, extns: Tuple) -> List:
        """
        Get list of files from dir according to extensions(extns)

        Parameters
        ----------
        pth : str
            Path ot dir
        extns: Tuple
            Set of file extensions
        """
        if not os.path.exists(pth):
            logging.warning(f"Dataset directory doesn't exist {pth}")
            return []
        files = [f for f in os.listdir(pth) if f.endswith(extns)]
        return files

    def __read_annotations(self, subset: int = None) -> pd.DataFrame:
        """
        Read annotations json

        Parameters
        ----------
        subset : int
            Length of subset for each target

        Returns
        -------
        pd.DataFrame
            Dataframe with annotations
        """
        exists_images = set()
        annotations_all = []

        for target in tqdm(self.conf.dataset.targets, desc=f"Prepare {self.dataset_type} dataset"):
            target_tsv = os.path.join(self.path_to_json, f"{target}.json")
            if os.path.exists(target_tsv):
                with open(target_tsv, "r") as file:
                    json_annotation = json.load(file)

                json_annotation = [
                    {**annotation, "name": f"{name}.jpg"} for name, annotation in json_annotation.items()
                ]
                if subset > 1:
                    json_annotation = json_annotation[:subset]

                annotation = pd.DataFrame(json_annotation)
                annotation["target"] = target
                annotations_all.append(annotation)
                exists_images.update(self.__get_files_from_dir(os.path.join(self.path_to_dataset, target), IMAGES))
            else:
                logging.info(f"Database for {target} not found")

        annotations_all = pd.concat(annotations_all, ignore_index=True)
        annotations_all["exists"] = annotations_all["name"].isin(exists_images)
        return annotations_all[annotations_all["exists"]]

    def __getitem__(self, item):
        """
        Get item from annotations
        """
        raise NotImplementedError

    def __len__(self):
        """
        Get length of dataset
        """
        return self.annotations.shape[0]


class DetectionDataset(HagridDataset):
    def __init__(self, conf: DictConfig, dataset_type: str, transform):
        """
        Parameters
        ----------
        conf : DictConfig
            Config for dataset
        dataset_type : str
            Type of dataset
        transform : albumentations.Compose
            Transformations for dataset
        """
        super().__init__(conf, dataset_type, transform)
        self.one_class = self.conf.dataset.get("one_class", False)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Dict]:
        """
        Get item from annotations

        Parameters
        ----------
        item : int
            Index of annotation item

        Returns
        -------
        Tuple[Image.Image, Dict]
            Image and target
        """
        row = self.annotations.iloc[[index]].to_dict("records")[0]

        image_pth = os.path.join(self.path_to_dataset, row["target"], row["name"])

        image = self._load_image(image_pth)

        if self.one_class:
            labels = np.array([self.labels[label] for label in row["labels"]])
            iter_boxes = [row["bboxes"][i] for i in range(len(row["bboxes"]))]
        else:
            labels = []
            iter_boxes = []
            for i in range(len(row["united_bbox"])):
                if row["united_bbox"][i] is None:
                    iter_boxes.append(row["bboxes"][i])
                    labels.append(self.labels[row["labels"][i]])
                else:
                    iter_boxes.append(row["united_bbox"][i])
                    labels.append(self.labels[row["united_label"][i]])
            labels = np.array(labels)

        target = {}
        width, height = image.size

        bboxes = []

        for bbox in iter_boxes:
            x1, y1, w, h = bbox
            x_min = x1 * width
            y_min = y1 * height
            x_max = (x1 + w) * width
            y_max = (y1 + h) * height
            if x_min < 0:
                x_min = 1
            if y_min > height:
                y_min = 1
            if x_max > width:
                x_max = width
            if y_max > height:
                y_max = height
            bboxes.append([x_min, y_min, x_max, y_max])

        bboxes = np.array(bboxes, dtype=np.float32)

        target["labels"] = labels
        target["boxes"] = bboxes

        image = np.array(image)
        if self.transform is not None:
            transformed_target = self.transform(image=image, bboxes=target["boxes"], class_labels=target["labels"])
            image = transformed_target["image"] / 255.0
            target["boxes"] = torch.tensor(transformed_target["bboxes"], dtype=torch.float32)
            target["labels"] = torch.tensor(transformed_target["class_labels"])

        if self.one_class:
            target["labels"] = torch.ones_like(target["labels"])

        return image, target


class ClassificationDataset(HagridDataset):
    def __init__(self, conf: DictConfig, dataset_type: str, transform):
        """
        Parameters
        ----------
        conf : DictConfig
            Config for dataset
        dataset_type : str
            Type of dataset
        transform : albumentations.Compose
            Transformations for dataset
        """
        super().__init__(conf, dataset_type, transform)
        self.annotations = self.annotations[
            ~self.annotations.apply(lambda x: x["labels"] == ["no_gesture"] and x["target"] != "no_gesture", axis=1)
        ]

        self.dataset_type = dataset_type

    def __getitem__(self, index: int) -> Tuple[Image.Image, Dict]:
        """
        Get item from annotations

        Parameters
        ----------
        index : int
            Index of annotation item

        Returns
        -------
        Tuple[Image.Image, Dict]
            Image and target
        """
        row = self.annotations.iloc[[index]].to_dict("records")[0]

        image_pth = os.path.join(self.path_to_dataset, row["target"], row["name"])

        image = self._load_image(image_pth)

        labels = row["labels"]

        if row["target"] == "no_gesture":
            gesture = "no_gesture"
        else:
            for label in labels:
                if label == "no_gesture":
                    continue
                else:
                    gesture = label
                    break
        try:
            label = {"labels": torch.tensor(self.labels[gesture])}
        except Exception:
            raise f"unknown gesture {gesture}"
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label
