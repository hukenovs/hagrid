"""Download models and datasets"""
import os
import argparse


urls = {
    "call": "https://sc.link/ykEn",
    "dislike": "https://sc.link/xjDB",
    "fist": "https://sc.link/wgB8",
    "four": "https://sc.link/vJA5",
    "like": "https://sc.link/r7wp",
    "mute": "https://sc.link/q8vp",
    "ok": "https://sc.link/pV0V",
    "one": "https://sc.link/oJqX",
    "palm": "https://sc.link/nJp7",
    "peace_inverted": "https://sc.link/mXoG",
    "peace": "https://sc.link/l6nM",
    "rock": "https://sc.link/kMm6",
    "stop_inverted": "https://sc.link/jJlv",
    "stop": "https://sc.link/gXgk",
    "three": "https://sc.link/wgBr",
    "three2": "https://sc.link/vJA8",
    "two_up_inverted": "https://sc.link/r7w2",
    "two_up": "https://sc.link/q8v7",
    "test": "https://sc.link/zlGy",
    "subsample": "https://sc.link/AO5l",
    "ann_train_val": "https://sc.link/BE5Y",
    "ann_test": "https://sc.link/DE5K",
    "ann_subsample": "https://sc.link/EQ5g"
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
    "two_up"
)

DATASET = os.path.expanduser("~/hagrid/dataset/")


def download(args):
    if args.subset:
        subsample = os.path.join(args.save_path, "subsample")
        os.makedirs(subsample, exist_ok=True)
        if args.dataset:
            os.system(f"wget {urls['subsample']} -O {subsample}/subsample.zip")
        if args.annotations:
            os.system(f"wget {urls['ann_subsample']} -O {subsample}/ann_subsample.zip")

    if args.test:
        testset = os.path.join(args.save_path, "test")
        os.makedirs(testset, exist_ok=True)
        if args.dataset:
            os.system(f"wget {urls['test']} -O {testset}/test.zip")
        if args.annotations:
            os.system(f"wget {urls['ann_test']} -O {testset}/ann_test.zip")

    if args.train:
        train_val = os.path.join(args.save_path, "train")
        os.makedirs(train_val, exist_ok=True)
        if args.dataset:
            for target in args.targets:
                os.system(f"wget {urls[target]} -O {train_val}/{target}.zip")
        if args.annotations:
            os.system(f"wget {urls['ann_train_val']} -O {train_val}/ann_train_val.zip")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download dataset...")
    parser.add_argument("--train", action='store_true', help="Download trainval set")
    parser.add_argument("--test", action='store_true', help="Download test set")
    parser.add_argument("--subset", action='store_true', help="Download subset with 100 items of each gesture")

    parser.add_argument("-a", "--annotations", action='store_true', help="Download annotations")
    parser.add_argument("-d", "--dataset", action='store_true', help="Download dataset")

    parser.add_argument("-t", "--targets", nargs="+", default=GESTURES, help="Target(s) for downloading train set")
    parser.add_argument("-p", "--save_path", type=str, default=DATASET, help="Save path")

    known_args, _ = parser.parse_known_args()
    return known_args


if __name__ == '__main__':
    params = parse_arguments()
    print("\n".join([f"{k :<30} : {v}" for k, v in vars(params).items()]))
    params.save_path = os.path.expanduser(params.save_path)
    download(params)
