import argparse
import json
import logging
import os
from typing import Tuple

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from converters.convert_utils import get_dataframe

tqdm.pandas()

logging.getLogger().setLevel(logging.INFO)


def get_area(bboxes: list) -> list:
    """
    Get area of bboxes
    Parameters
    ----------
    bboxes: list
        list of bboxes

    Returns
    -------
    list
    """
    bboxes = np.array(bboxes)
    area = bboxes[:, 2] * bboxes[:, 3]
    return area


def get_w_h(img_path: str) -> Tuple[int, int]:
    """
    Get width and height of image
    Parameters
    ----------
    img_path: str
        path to image

    Returns
    -------
    Tuple[int, int]
    """
    img = Image.open(img_path)
    img_w, img_h = img.size
    return img_w, img_h


def get_abs_bboxes(bboxes: list, im_size: tuple) -> list:
    """
    Get absolute bboxes in format [xmin, ymin, w, h]
    Parameters
    ----------
    bboxes: list
        list of bboxes
    im_size: tuple
        image size

    Returns
    -------
    list
    """
    width, height = im_size
    bboxes_out = []
    for box in bboxes:
        x1, y1, w, h = box
        bbox_abs = [x1 * width, y1 * height, w * width, h * height]
        bboxes_out.append(bbox_abs)
    return bboxes_out


def get_poly(bboxes: list) -> list:
    """
    Get polygon from bboxes
    Parameters
    ----------
    bboxes: list
        list of bboxes

    Returns
    -------
    list
    """
    poly = []
    for box in bboxes:
        xmin, ymin, w, h = box
        xmax = xmin + w
        ymax = ymin + h
        poly.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])

    return poly


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
    labels = {label: num for (label, num) in zip(conf.dataset.targets, range(len(conf.dataset.targets)))}
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    dataset_folder = conf.dataset.dataset_folder
    phases = conf.dataset.phases
    for phase in phases:
        logging.info(f"Run convert {phase}")
        logging.info("Create Dataframe")
        annotations = get_dataframe(conf, phase)

        logging.info("Create image_path")
        annotations["image_path"] = annotations.progress_apply(
            lambda row: os.path.join(dataset_folder, row["target"], row["name"]), axis=1
        )

        logging.info("Create width, height")
        w_h = annotations["image_path"].progress_apply(lambda x: get_w_h(x))
        annotations["width"] = np.array(w_h.to_list())[:, 0]
        annotations["height"] = np.array(w_h.to_list())[:, 1]

        logging.info("Create id")
        annotations["id"] = annotations.index

        logging.info("Create abs_bboxes")
        annotations["abs_bboxes"] = annotations.progress_apply(
            lambda row: get_abs_bboxes(row["bboxes"], (row["width"], row["height"])), axis=1
        )
        logging.info("Create area")
        annotations["area"] = annotations["abs_bboxes"].progress_apply(lambda bboxes: get_area(bboxes))
        logging.info("Create segmentation")
        annotations["segmentation"] = annotations["abs_bboxes"].progress_apply(lambda bboxes: get_poly(bboxes))
        logging.info("Create category_id")
        annotations["category_id"] = annotations["labels"].progress_apply(lambda x: [labels[label] for label in x])

        categories = [{"supercategory": "none", "name": k, "id": v} for k, v in labels.items()]
        logging.info(f"Save to {phase}.json")
        res_file = {"categories": categories, "images": [], "annotations": []}
        annot_count = 0
        for index, row in tqdm(annotations.iterrows()):
            img_elem = {"file_name": row["image_path"], "height": row["height"], "width": row["width"], "id": row["id"]}

            res_file["images"].append(img_elem)

            num_boxes = len(row["bboxes"])
            for i in range(num_boxes):
                annot_elem = {
                    "id": annot_count,
                    "bbox": row["abs_bboxes"][i],
                    "segmentation": [row["segmentation"][i]],
                    "image_id": row["id"],
                    "category_id": row["category_id"][i],
                    "iscrowd": 0,
                    "area": row["area"][i],
                }
                res_file["annotations"].append(annot_elem)
                annot_count += 1

        with open(f"{args.out}/{phase}.json", "w") as f:
            json_str = json.dumps(res_file)
            f.write(json_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert Hagrid annotations to Coco annotations format", add_help=False)
    parser.add_argument("--cfg", default="converter_config.yaml", type=str, help="path to data config")
    parser.add_argument("--out", default="./hagrid_coco_format", type=str, help="path to output jsons")
    args = parser.parse_args()
    run_convert(args)
