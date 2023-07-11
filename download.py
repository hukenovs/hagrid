"""Download models and datasets"""
import argparse
import os

main_url = "https://n-usr-2uzac.s3pd12.sbercloud.ru/b-usr-2uzac-mv4/hagrid/"
urls = {
    "call": f"{main_url}train_val_call.zip",
    "dislike": f"{main_url}train_val_dislike.zip",
    "fist": f"{main_url}train_val_fist.zip",
    "four": f"{main_url}train_val_four.zip",
    "like": f"{main_url}train_val_like.zip",
    "mute": f"{main_url}train_val_mute.zip",
    "ok": f"{main_url}train_val_ok.zip",
    "one": f"{main_url}train_val_one.zip",
    "palm": f"{main_url}train_val_palm.zip",
    "peace_inverted": f"{main_url}train_val_peace_inverted.zip",
    "peace": f"{main_url}train_val_peace.zip",
    "rock": f"{main_url}train_val_rock.zip",
    "stop_inverted": f"{main_url}train_val_stop_inverted.zip",
    "stop": f"{main_url}train_val_stop.zip",
    "three": f"{main_url}train_val_three.zip",
    "three2": f"{main_url}train_val_three2.zip",
    "two_up_inverted": f"{main_url}train_val_two_up_inverted.zip",
    "two_up": f"{main_url}train_val_two_up.zip",
    "test": f"{main_url}test.zip",
    "subsample": f"{main_url}subsample.zip",
    "ann_train_val": f"{main_url}ann_train_val.zip",
    "ann_test": f"{main_url}ann_test.zip",
    "ann_subsample": f"{main_url}ann_subsample.zip",
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
    parser.add_argument("--train", action="store_true", help="Download trainval set")
    parser.add_argument("--test", action="store_true", help="Download test set")
    parser.add_argument("--subset", action="store_true", help="Download subset with 100 items of each gesture")

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
