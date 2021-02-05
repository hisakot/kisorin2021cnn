import csv
import cv2
import glob
import numpy as np
import os

import torch
from torch.utils.data import Dataset

DATASET_CACHE = "./dataset_cache"
ROOT_DIR = "./flowers/"
W = 224
H = 224

class Datas():
    def __init__(self):
        self.dataset = list()
        self.length = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # load img
        img_path = self.dataset[idx]["img_path"]
        image = cv2.imread(img_path) # (h, w, c)
        image = cv2.resize(image, (W, H))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1)) # (c, h, w)
        image = torch.tensor(image, dtype=torch.float32) # tensor

        # load label
        label = self.dataset[idx]["label"] # (x, y)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

def make_dataset():
    dataset_dicts = list()
    img_dirs = os.listdir(ROOT_DIR)
    img_dirs_list = [ROOT_DIR + d + "/*.jpg" for d in img_dirs]
    img_dirs_list.sort()
    for i, img_dir in enumerate(img_dirs_list):
        img_paths = glob.glob(img_dir)
        img_paths.sort()
        for img_path in img_paths:
            dataset_dicts.append({"img_path" : img_path,
                                  "label" : i,})

    return dataset_dicts


def setup_data():
    datas = Datas()

    try:
        cache = torch.load(DATASET_CACHE)
        datas.dataset = cache["dataset"]
        datas.length = cache["length"]

    except FileNotFoundError:
        dataset_dicts = make_dataset()
        datas.dataset = dataset_dicts
        datas.length = len(datas.dataset)

        cache_dict = {"dataset" : datas.dataset,
                      "length" : datas.length,}
        torch.save(cache_dict, DATASET_CACHE)

    return datas
