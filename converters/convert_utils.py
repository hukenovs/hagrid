import json
import logging
import os
from typing import Union

import pandas as pd
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

from constants import IMAGES


def get_files_from_dir(pth, extns):
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
    np.array
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
            exists_images.extend(get_files_from_dir(os.path.join(dataset_folder, target), IMAGES))
        else:
            logging.warning(f"Database for {target} not found")

    annotations_all["exists"] = annotations_all["name"].isin(exists_images)
    annotations = annotations_all[annotations_all["exists"]]

    return annotations
