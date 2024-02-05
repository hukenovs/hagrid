# HaGRID - HAnd Gesture Recognition Image Dataset

![hagrid](images/hagrid.jpg)

We introduce a large image dataset **HaGRID** (**HA**nd **G**esture **R**ecognition **I**mage **D**ataset) for hand gesture recognition (HGR) systems. You can use it for image classification or image detection tasks. Proposed dataset allows to build HGR systems, which can be used in video conferencing services (Zoom, Skype, Discord, Jazz etc.), home automation systems, the automotive sector, etc.

HaGRID size is **723GB** and dataset contains **554,800** FullHD RGB images divided into **18** classes of gestures. Also, some images have `no_gesture` class if there is a second free hand in the frame. This extra class contains **120,105** samples. The data were split into training 74%, 10% validation and testing 16% sets by subject `user_id`, with 410,800 images for train, 54,000 images for validation and 90,000 for test.

![gestures](images/gestures.jpg)

The dataset contains **37,583** unique persons and at least this number of unique scenes. The subjects are people from 18 to 65 years old. The dataset was collected mainly indoors with considerable variation in lighting, including artificial and natural light. Besides, the dataset includes images taken in extreme conditions such as facing and backing to a window. Also, the subjects had to show gestures at a distance of 0.5 to 4 meters from the camera.

Example of sample and its annotation:

![example](images/example.jpeg)

For more information see our arxiv paper [HaGRID - HAnd Gesture Recognition Image Dataset](https://arxiv.org/abs/2206.08219).

## 🔥 Changelog
- **`2023/09/21`**: We release HaGRID 2.0. ✌️
  - All files for training and testing are combined into one directory
  - The data was further cleared and new ones were added
  - Multi-gpu training and testing
  - Added new models for detection and full-frame classification
  - Dataset size is **723GB**
  - **554,800** FullHD RGB images (cleaned and updated classes, added diversity by race)
  - Extra class `no_gesture` contains **120,105** samples
  - Train/val/test split: (410,800) **74%** / (54,000) **10%** / (90,000) **16%** by subject `user_id`
  - **37,583** unique persons
- **`2022/06/16`**: HaGRID (Initial Dataset) 💪
  - Dataset size is **716GB**
  - **552,992** FullHD RGB images divided into **18** classes
  - Extra class `no_gesture` contains **123,589** samples
  - Train/test split: (509,323) **92%** / (43,669) **8%** by subject `user_id`
  - **34,730** unique persons from 18 to 65 years old
  - The distance is 0.5 to 4 meters from the camera

Old HaGRID dataset is also available into branch `hagrid_v1`!

## Installation
Clone and install required python packages:
```bash
git clone https://github.com/hukenovs/hagrid.git
# or mirror link:
cd hagrid
# Create virtual env by conda or venv
conda create -n gestures python=3.11 -y
conda activate gestures
# Install requirements
pip install -r requirements.txt
```

## Downloads
We split the train dataset into 18 archives by gestures because of the large size of data. Download and unzip them from the following links:

### Dataset

| Gesture                           | Size    | Gesture                                   | Size    |
|-----------------------------------|---------|-------------------------------------------|---------|
| [`call`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/call.zip)    | 37.2 GB | [`peace`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/peace.zip)           | 41.4 GB |
| [`dislike`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/dislike.zip) | 40.9 GB | [`peace_inverted`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/peace_inverted.zip)  | 40.5 GB |
| [`fist`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/fist.zip)    | 42.3 GB | [`rock`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/rock.zip)            | 41.7 GB |
| [`four`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/four.zip)    | 43.1 GB | [`stop`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/stop.zip)            | 41.8 GB |
| [`like`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/like.zip)    | 42.2 GB | [`stop_inverted`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/stop_inverted.zip)   | 41.4 GB |
| [`mute`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/mute.zip)    | 43.2 GB | [`three`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/three.zip)           | 42.2 GB |
| [`ok`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/ok.zip)      | 42.5 GB | [`three2`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/three2.zip)          | 40.2 GB |
| [`one`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/one.zip)     | 42.7 GB | [`two_up`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/two_up.zip)          | 41.8 GB |
| [`palm`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/palm.zip)    | 43.0 GB | [`two_up_inverted`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset/two_up_inverted.zip) | 40.9 GB |

`dataset` **annotations**: [`annotations`](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/annotations.zip)

[HaGRID 512px - lightweight version of the full dataset with](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_dataset_new_554800/hagrid_dataset_512.zip) `min_side = 512p` `26.4 GB`

or by using python script
```bash
python download.py --save_path <PATH_TO_SAVE> \
                   --annotations \
                   --dataset
```

Run the following command with key `--dataset` to download dataset with images. Download annotations for selected stage by `--annotations` key.

```bash
usage: download.py [-h] [-a] [-d] [-t TARGETS [TARGETS ...]] [-p SAVE_PATH]

Download dataset...

optional arguments:
  -h, --help            show this help message and exit
  -a, --annotations     Download annotations
  -d, --dataset         Download dataset
  -t TARGETS [TARGETS ...], --targets TARGETS [TARGETS ...]
                        Target(s) for downloading train set
  -p SAVE_PATH, --save_path SAVE_PATH
                        Save path
```
After downloading, you can unzip the archive by running the following command:
```bash
unzip <PATH_TO_ARCHIVE> -d <PATH_TO_SAVE>
```
The structure of the dataset is as follows:
```
├── hagrid_dataset <PATH_TO_DATASET_FOLDER>
│   ├── call
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── ...
├── hagrid_annotations
│   ├── train <PATH_TO_JSON_TRAIN>
│   │   ├── call.json
│   │   ├── ...
│   ├── val <PATH_TO_JSON_VAL>
│   │   ├── call.json
│   │   ├── ...
│   ├── test <PATH_TO_JSON_TEST>
│   │   ├── call.json
│   │   ├── ...
```

## Models
We provide some pre-trained models as the baseline with the classic backbone architectures for gesture classification and gesture detection.

| Detector                                         | mAP      |
|--------------------------------------------------|----------|
| [SSDLiteMobileNetV3Small](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_models_new/SSDLite_MobilenetV3_small.pth)  | 57.7     |
| [SSDLiteMobileNetV3Large](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_models_new/SSDLite_MobilenetV3_large.pth)  | 71.6     |
| [RetinaNet_ResNet50](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_models_new/RetinaNet_ResNet50.pth) | **79.1** |
| [YoloV7Tiny](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_models_new/YoloV7Tiny.onnx)               | 71.6     |


However, if you need a single gesture, you can use pre-trained full frame classifiers instead of detectors.
To use full frame models, **remove the no_gesture class**

| Full Frame Classifiers                    | F1 Gestures |
|-------------------------------------------|---------|
| [MobileNetV3_small](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_models_new/MobileNetV3_small.pth) | 86.4    |
| [MobileNetV3_large](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_models_new/MobileNetV3_large.pth) | 91.9    |
| [VitB16](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_models_new/VitB16.pth) | 91.1    |
| [ResNet18](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_models_new/ResNet18.pth)      | 97.5    |
| [ResNet152](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_models_new/ResNet152.pth)    | 95.5    |
| [ResNeXt50](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_models_new/ResNext50.pth)    | **98.3** |
| [ResNeXt101](https://n-ws-620xz-pd11.s3pd11.sbercloud.ru/b-ws-620xz-pd11-jux/hagrid/hagrid_models_new/ResNext101.pth)  | 97.5    |

<details><summary><h3>Train</h3></summary>

You can use downloaded trained models, otherwise select a parameters for training in `configs` folder.
To train the model, execute the following command:

Single GPU:

```bash
python run.py -c train -p configs/<config>
```
Multi GPU:
```bash
bash ddp_run.sh -g 0,1,2,3 -c train -p configs/<config>
```
which -g is a list of GPU ids.


Every step, the current loss, learning rate and others values get logged to **Tensorboard**.
See all saved metrics and parameters by opening a command line (this will open a webpage at `localhost:6006`):
```bash
tensorboard --logdir=<workdir>
```
</details>
<details><summary><h3>Test</h3></summary>

Test your model by running the following command:

Single GPU:

```bash
python run.py -c test -p configs/<config>
```
Multi GPU:
```bash
bash ddp_run.sh -g 0,1,2,3 -c test -p configs/<config>
```
which -g is a list of GPU ids.
</details>

## Demo
 ```bash
python demo.py -p <PATH_TO_CONFIG> --landmarks
```
![demo](images/demo.gif)

## Demo Full Frame Classifiers
 ```bash
python demo_ff.py -p <PATH_TO_CONFIG>
```

## Annotations

The annotations consist of bounding boxes of hands in COCO format `[top left X position, top left Y position, width, height]` with gesture labels. We provide `user_id` field that will allow you to split the train / val / test dataset yourself.
```json
"0534147c-4548-4ab4-9a8c-f297b43e8ffb": {
  "bboxes": [
    [0.38038597, 0.74085361, 0.08349486, 0.09142549],
    [0.67322755, 0.37933984, 0.06350809, 0.09187757]
  ],
  "labels": [
    "no_gesture",
    "one"
  ],
  "user_id": "bb138d5db200f29385f..."
}
```
- Key - image name without extension
- Bboxes - list of normalized bboxes `[top left X pos, top left Y pos, width, height]`
- Labels - list of class labels e.g. `like`, `stop`, `no_gesture`
- User ID - subject id (useful for split data to train / val subsets).

### Bounding boxes

| Object       | Train + Val | Test    | Total   |
|--------------|-------------|---------|---------|
| gesture      | ~ 28 300    | ~ 2 400 | 30 629  |
| no gesture   | 112 740     | 10 849  | 123 589 |
| total boxes  | 622 063     | 54 518  | 676 581 |

### Converters

<details><summary> <b>Yolo</b> </summary>

We provide a script to convert annotations to [YOLO](https://pjreddie.com/darknet/yolo/) format. To convert annotations, run the following command:

```bash
python -m converters.hagrid_to_yolo --path_to_config <PATH>
```

after conversion, you need change original definition [img2labels](https://github.com/WongKinYiu/yolov7/blob/2fdc7f14395f6532ad05fb3e6970150a6a83d290/utils/datasets.py#L347-L350) to:

```python
def img2label_paths(img_paths):
    img_paths = list(img_paths)
    # Define label paths as a function of image paths
    if "train" in img_paths[0]:
        return [x.replace("train", "train_labels").replace(".jpg", ".txt") for x in img_paths]
    elif "test" in img_paths[0]:
        return [x.replace("test", "test_labels").replace(".jpg", ".txt") for x in img_paths]
    elif "val" in img_paths[0]:
        return [x.replace("val", "val_labels").replace(".jpg", ".txt") for x in img_paths]
```

</details>


<details><summary> <b>Coco</b> </summary>

Also, we provide a script to convert annotations to [Coco](https://cocodataset.org/#home) format. To convert annotations, run the following command:

```bash
python -m converters.hagrid_to_coco --path_to_config <PATH>
```

</details>

### License
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a variant of <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

Please see the specific [license](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf).

### Authors and Credits
- [Alexander Kapitanov](https://www.linkedin.com/in/hukenovs)
- [Andrey Makhlyarchuk](https://www.linkedin.com/in/makhliarchuk)
- [Karina Kvanchiani](https://www.linkedin.com/in/kvanchiani)
- [Aleksandr Nagaev](https://www.linkedin.com/in/nagadit)
- [Roman Kraynov](https://ru.linkedin.com/in/roman-kraynov-25ab44265)

### Links
- [Github](https://github.com/hukenovs/hagrid)
- [Mirror](https://gitlab.aicloud.sbercloud.ru/rndcv/hagrid)
- [arXiv](https://arxiv.org/abs/2206.08219)
- [Kaggle](https://www.kaggle.com/datasets/kapitanov/hagrid)
- [Habr](https://habr.com/ru/company/sberdevices/blog/671614/)
- [Paperswithcode](https://paperswithcode.com/paper/hagrid-hand-gesture-recognition-image-dataset)

### Citation
You can cite the paper using the following BibTeX entry:

    @InProceedings{Kapitanov_2024_WACV,
        author    = {Kapitanov, Alexander and Kvanchiani, Karina and Nagaev, Alexander and Kraynov, Roman and Makhliarchuk, Andrei},
        title     = {HaGRID -- HAnd Gesture Recognition Image Dataset},
        booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
        month     = {January},
        year      = {2024},
        pages     = {4572-4581}
    }
