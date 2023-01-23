import argparse
import json
import logging
import os
from typing import Tuple, Union

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from constants import IMAGES

logging.getLogger().setLevel(logging.INFO)

tqdm.pandas()


def xywh_to_cxcywh(boxes):
    """
    Convert xywh to cxcywh
    Parameters
    ----------
    boxes: list
        list of bboxes

    Returns
    -------
    list
    """
    boxes = np.array(boxes)
    return np.concatenate([boxes[..., :2] + boxes[..., 2:] / 2, boxes[..., 2:]], len(boxes.shape) - 1)


def xywh_to_xyxy(boxes):
    """
    Convert xywh to xyxy
    Parameters
    ----------
    boxes: list
        list of bboxes

    Returns
    -------
    list
    """
    boxes = np.array(boxes)
    return np.concatenate([boxes[..., :2], boxes[..., :2] + boxes[..., 2:]], len(boxes.shape) - 1)


def get_files_from_dir(pth: str, extns: Tuple) -> list:
    """
    Get files from directory
    Parameters
    ----------
    pth: str
        path to directory
    extns: Tuple
        extensions of files

    Returns
    -------
    list
    """
    if not os.path.exists(pth):
        logging.error(f"Dataset directory doesn't exist {pth}")
        return []
    files = [f for f in os.listdir(pth) if f.endswith(extns)]
    return files


def get_dataframe(conf: Union[DictConfig, ListConfig], phase: str) -> pd.DataFrame:
    """
    Get dataframe with annotations
    Parameters
    ----------
    conf: Union[DictConfig, ListConfig]
        config
    phase: str
        phase of dataset

    Returns
    -------
    pd.DataFrame
    """
    dataset_annotations = conf.dataset.dataset_annotations
    dataset_folder = conf.dataset.dataset_folder
    targets = conf.dataset.targets
    annotations_all = None
    exists_images = []

    for target in tqdm(targets):
        target_json = os.path.join(dataset_annotations, f"{phase}", f"{target}.json")
        if os.path.exists(target_json):
            json_annotation = json.load(open(os.path.join(target_json)))

            json_annotation = [
                dict(annotation, **{"name": f"{name}.jpg"})
                for name, annotation in zip(json_annotation, json_annotation.values())
            ]

            annotation = pd.DataFrame(json_annotation)

            annotation["target"] = target
            annotations_all = pd.concat([annotations_all, annotation], ignore_index=True)
            exists_images.extend(get_files_from_dir(os.path.join(dataset_folder, phase, target), IMAGES))
        else:
            logging.warning(f"Database for {target} not found")

    annotations_all["exists"] = annotations_all["name"].isin(exists_images)
    annotations = annotations_all[annotations_all["exists"]]

    return annotations


def run_convert(args: argparse.Namespace) -> None:
    """
    Run convert
    Parameters
    ----------
    args: argparse.Namespace
        arguments

    Returns
    -------
    None
    """
    conf = OmegaConf.load(args.cfg)
    bbox_format = args.bbox_format
    labels = {label: num for (label, num) in zip(conf.dataset.targets, range(len(conf.dataset.targets)))}

    dataset_folder = conf.dataset.dataset_folder
    phases = conf.dataset.phases
    for phase in phases:
        phase_dir = os.path.join(dataset_folder, f"{phase}_labels")
        if not os.path.exists(phase_dir):
            os.mkdir(phase_dir)

        logging.info(f"Run convert {phase}")
        logging.info("Create Dataframe")
        annotations = get_dataframe(conf, phase)

        logging.info("Create image_path")
        annotations["image_path"] = annotations.progress_apply(
            lambda row: os.path.join(dataset_folder, phase, row["target"], row["name"]), axis=1
        )

        if bbox_format == "cxcywh":
            logging.info("Create bboxes cxcywh format")
            annotations["converted_bboxes"] = annotations.progress_apply(
                lambda row: xywh_to_cxcywh(row["bboxes"]), axis=1
            )

        elif bbox_format == "xyxy":
            logging.info("Create bboxes xyxy format")
            annotations["converted_bboxes"] = annotations.progress_apply(
                lambda row: xywh_to_xyxy(row["bboxes"]), axis=1
            )

        elif bbox_format == "xywh":
            logging.info("Create bboxes xywh format")
            annotations["converted_bboxes"] = annotations["bboxes"]

        else:
            assert False, f"Unknown bbox format {bbox_format}"

        logging.info("Create labels_id")
        annotations["category_id"] = annotations["labels"].progress_apply(lambda x: [labels[label] for label in x])

        logging.info("Convert")
        for target in annotations["target"].unique():
            label_dir = os.path.join(phase_dir, target)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)

            temp_annotations = annotations[annotations["target"] == target]
            for row_id in tqdm(range(len(temp_annotations)), desc=target):
                row = temp_annotations.iloc[row_id]
                label_path = row["image_path"].replace(phase, f"{phase}_labels").replace(".jpg", ".txt")
                with open(label_path, "w") as f:
                    for i in range(len(row["labels"])):
                        f.write(str(row["category_id"][i]) + " ")
                        f.write(" ".join(map(str, row["converted_bboxes"][i])) + "\n")

        with open(f"{phase}.txt", "w") as f:
            for img_path in annotations["image_path"]:
                f.write(f"{img_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert Hagrid annotations to Yolo annotations format", add_help=False)
    parser.add_argument("--bbox_format", default="cxcywh", type=str, help="bbox format: xyxy, cxcywh, xywh")
    parser.add_argument("--cfg", default="converter_config.yaml", type=str, help="path to data config")
    args = parser.parse_args()
    run_convert(args)
