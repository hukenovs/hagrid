"""Download models and datasets"""

import argparse
import os

v2_url = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/"
main_url = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/"

urls = {
    "call": f"{main_url}hagrid_dataset/call.zip",
    "dislike": f"{main_url}hagrid_dataset/dislike.zip",
    "fist": f"{main_url}hagrid_dataset/fist.zip",
    "four": f"{main_url}hagrid_dataset/four.zip",
    "like": f"{main_url}hagrid_dataset/like.zip",
    "mute": f"{main_url}hagrid_dataset/mute.zip",
    "ok": f"{main_url}hagrid_dataset/ok.zip",
    "one": f"{v2_url}hagrid_v2_zip/one.zip",
    "palm": f"{main_url}hagrid_dataset/palm.zip",
    "peace_inverted": f"{main_url}hagrid_dataset/peace_inverted.zip",
    "peace": f"{main_url}hagrid_dataset/peace.zip",
    "rock": f"{v2_url}hagrid_v2_zip/rock.zip",
    "stop_inverted": f"{main_url}hagrid_dataset/stop_inverted.zip",
    "stop": f"{main_url}hagrid_dataset/stop.zip",
    "three": f"{main_url}hagrid_dataset/three.zip",
    "three2": f"{main_url}hagrid_dataset/three2.zip",
    "two_up_inverted": f"{main_url}hagrid_dataset/two_up_inverted.zip",
    "two_up": f"{main_url}hagrid_dataset/two_up.zip",
    "grabbing": f"{v2_url}hagrid_v2_zip/grabbing.zip",
    "grip": f"{v2_url}hagrid_v2_zip/grip.zip",
    "holy": f"{v2_url}hagrid_v2_zip/holy.zip",
    "point": f"{v2_url}hagrid_v2_zip/point.zip",
    "three3": f"{v2_url}hagrid_v2_zip/three3.zip",
    "timeout": f"{v2_url}hagrid_v2_zip/timeout.zip",
    "xsign": f"{v2_url}hagrid_v2_zip/xsign.zip",
    "hand_heart": f"{v2_url}hagrid_v2_zip/hand_heart.zip",
    "hands_heart2": f"{v2_url}hagrid_v2_zip/hands_heart2.zip",
    "little_finger": f"{v2_url}hagrid_v2_zip/little_finger.zip",
    "middle_finger": f"{v2_url}hagrid_v2_zip/middle_finger.zip",
    "take_picture": f"{v2_url}hagrid_v2_zip/take_picture.zip",
    "three_gun": f"{v2_url}hagrid_v2_zip/three_gun.zip",
    "thumb_index": f"{v2_url}hagrid_v2_zip/thumb_index.zip",
    "thumb_index2": f"{v2_url}hagrid_v2_zip/thumb_index2.zip",
    "no_gesture": f"{v2_url}hagrid_v2_zip/no_gesture.zip",
    "annotations": f"{v2_url}/annotations_with_landmarks/annotations.zip",
}

GESTURES = (
    "call",
    "dislike",
    "fist",
    "four",
    "like",
    "mute",
    "ok",
    "one",
    "palm",
    "peace_inverted",
    "peace",
    "rock",
    "stop_inverted",
    "stop",
    "three",
    "three2",
    "two_up_inverted",
    "two_up",
    "grabbing",
    "grip",
    "holy",
    "point",
    "three3",
    "timeout",
    "xsign",
    "hand_heart",
    "hands_heart2",
    "little_finger",
    "middle_finger",
    "take_picture",
    "three_gun",
    "thumb_index",
    "thumb_index2",
    "no_gesture",
)

DATASET = os.path.expanduser("./hagrid/dataset/")


def download(args):
    dataset = args.save_path
    os.makedirs(dataset, exist_ok=True)
    if args.dataset:
        for target in args.targets:
            os.system(f"wget {urls[target]} -O {dataset}/{target}.zip")
    if args.annotations:
        os.system(f"wget {urls['annotations']} -O {dataset}/ann.zip")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download dataset...")

    parser.add_argument("-a", "--annotations", action="store_true", help="Download annotations")
    parser.add_argument("-d", "--dataset", action="store_true", help="Download dataset")

    parser.add_argument("-t", "--targets", nargs="+", default=GESTURES, help="Target(s) for downloading train set")
    parser.add_argument("-p", "--save_path", type=str, default=DATASET, help="Save path")

    known_args, _ = parser.parse_known_args()
    return known_args


if __name__ == "__main__":
    params = parse_arguments()
    print("\n".join([f"{k :<30} : {v}" for k, v in vars(params).items()]))
    params.save_path = os.path.expanduser(params.save_path)
    download(params)
