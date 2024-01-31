"""Download models and datasets"""
import argparse
import os

main_url = "https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/"
urls = {
    "call": f"{main_url}hagrid_dataset/call.zip",
    "dislike": f"{main_url}hagrid_dataset/dislike.zip",
    "fist": f"{main_url}hagrid_dataset/fist.zip",
    "four": f"{main_url}hagrid_dataset/four.zip",
    "like": f"{main_url}hagrid_dataset/like.zip",
    "mute": f"{main_url}hagrid_dataset/mute.zip",
    "ok": f"{main_url}hagrid_dataset/ok.zip",
    "one": f"{main_url}hagrid_dataset/one.zip",
    "palm": f"{main_url}hagrid_dataset/palm.zip",
    "peace_inverted": f"{main_url}hagrid_dataset/peace_inverted.zip",
    "peace": f"{main_url}hagrid_dataset/peace.zip",
    "rock": f"{main_url}hagrid_dataset/rock.zip",
    "stop_inverted": f"{main_url}hagrid_dataset/stop_inverted.zip",
    "stop": f"{main_url}hagrid_dataset/stop.zip",
    "three": f"{main_url}hagrid_dataset/three.zip",
    "three2": f"{main_url}hagrid_dataset/three2.zip",
    "two_up_inverted": f"{main_url}hagrid_dataset/two_up_inverted.zip",
    "two_up": f"{main_url}hagrid_dataset/two_up.zip",
    "annotations": f"{main_url}annotations.zip",
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
