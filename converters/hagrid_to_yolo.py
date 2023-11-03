import argparse
import logging
import os

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from converters.convert_utils import get_dataframe

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
    np.array
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
    np.array
    """
    boxes = np.array(boxes)
    return np.concatenate([boxes[..., :2], boxes[..., :2] + boxes[..., 2:]], len(boxes.shape) - 1)


def create_label(row):
    """
    Create .txt files with bboxes and labels
    Parameters
    ----------
    row:
     row in annotation DataFrame
    Returns
    -------
    None
    """
    with open(row["label_path"], "w") as f:
        for i in range(len(row["labels"])):
            f.write(str(row["category_id"][i]) + " ")
            f.write(" ".join(map(str, row["converted_bboxes"][i])) + "\n")


def create_hardlink(row, current_dir, dataset_folder, phase):
    """
    Create hardlink of HaGRID images
    Parameters
    ----------
    row:
     row in annotation DataFrame
    current_dir:
     path to dataset after converting
    dataset_folder:
     path to HaGRID dataset
    phase:
     currently processed subsample
    """
    if not os.path.exists(os.path.join(current_dir, phase, row["target"])):
        os.mkdir(os.path.join(current_dir, phase, row["target"]))
    os.link(
        os.path.join(dataset_folder, row["target"], row["name"]),
        os.path.join(current_dir, phase, row["target"], row["name"]),
    )


def run_convert(args):
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
    out_dir = os.path.abspath(args.out)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for phase in phases:
        phase_dir = os.path.join(out_dir, f"{phase}_labels")
        if not os.path.exists(phase_dir):
            os.mkdir(phase_dir)

        logging.info(f"Run convert {phase}")
        logging.info("Create Dataframe")
        annotations = get_dataframe(conf, phase)

        logging.info(f"Create hardlinks for {phase} data")
        if not os.path.exists(os.path.join(out_dir, phase)):
            os.mkdir(os.path.join(out_dir, phase))
        annotations.progress_apply(lambda row: create_hardlink(row, out_dir, dataset_folder, phase), axis=1)

        logging.info("Create image_path")
        annotations["image_path"] = annotations.progress_apply(
            lambda row: os.path.join(out_dir, phase, row["target"], row["name"]), axis=1
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

        logging.info("Create labels_id")
        annotations["category_id"] = annotations["labels"].progress_apply(lambda x: [labels[label] for label in x])
        annotations["label_path"] = annotations["image_path"].progress_apply(
            lambda x: x.replace(phase, f"{phase}_labels").replace(".jpg", ".txt")
        )
        logging.info("Convert")
        for target in annotations["target"].unique():
            logging.info(f"Convert {target}")
            label_dir = os.path.join(phase_dir, target)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)

            temp_annotations = annotations[annotations["target"] == target]
            _ = temp_annotations.progress_apply(lambda x: create_label(x), axis=1)
        with open(os.path.join(out_dir, f"{phase}.txt"), "w") as f:
            for img_path in annotations["image_path"]:
                f.write(f"{img_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Convert Hagrid annotations to Yolo annotations format", add_help=False)
    parser.add_argument("--bbox_format", default="cxcywh", type=str, help="bbox format: xyxy, cxcywh, xywh")
    parser.add_argument("--cfg", default="converter_config.yaml", type=str, help="path to data config")
    parser.add_argument("--out", default="./hagrid_yolo_format", type=str, help="path to output dir")
    args = parser.parse_args()
    run_convert(args)
