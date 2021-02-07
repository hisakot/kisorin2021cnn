import cv2
import glob
import numpy as np
import os

import torch

import common
import blend
import model

W = 224
H = 224

MODEL_SAVE_PATH = "./models/"
IMG_PATH = "./flowers/"

def forward(model, device, img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, (W, H))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1)) # (c, h, w)
    # make single batch tensor
    image = torch.tensor(image[np.newaxis, :, :], dtype=torch.int64)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)

    return output

if __name__ == '__main__':
    # setup model and GPU
    model = model.AlexNet(pretrained=False, out_classes=5)
    model, device = common.setup_device(model)

    save_model = glob.glob(MODEL_SAVE_PATH + "*")[0]
    checkpoint = torch.load(save_model, map_location=device)
    if torch.cuda.device_count() >= 1:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:] # remove "module."
            state_dict[name] = v
        model.load_state_dict(state_dict)
    model.eval()

    # main
    img_paths = glob.glob(IMG_PATH)
    img_paths.sort()

    if not os.path.exists(INF_GAZE_CSV):
        print("========== forward and save inferenced result ==========")
        for i, img_path in enumerate(img_paths):
            # inference
            output = forward(model, device, img_path)
            output = output.cpu().numpy()
            result = np.array([img_path, output])

            # save inferenced gaze as txt
            with open (INF_CSV, "a") as f:
                np.savetxt(f, result, delimiter=",")