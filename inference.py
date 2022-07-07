#!/usr/bin/env python
# coding: utf-8

import sys
from goat.engine import FitterMaskRCNN
from goat.model import maskRCNNModel, predict_image
from goat.dataset import InferenceMaskRCNNDataset
from goat.utils import contours_to_eccentricity, get_pixel_area_in_percent_from_contours, brightness_by_contours
import glob
import cv2
import numpy as np
import pandas as pd
import os

from config import Config
config = Config()


def inference(IMAGE_FOLDER, OUTPUT_FOLDER):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    ext = ['png', 'jpg', 'tif']  # Add image formats here
    img_paths = []
    [img_paths.extend(glob.glob(IMAGE_FOLDER + '/*.' + e)) for e in ext]
    print(f"Found {len(img_paths)} images")

    model = maskRCNNModel()
    fitter = FitterMaskRCNN(model, config.device, config)
    fitter.load(config.model_weights)

    inference_ds = InferenceMaskRCNNDataset(img_paths, config.resize)

    results = []
    for i, (img, meta) in enumerate(inference_ds):
        print(meta["path"])
        loadgreyscale = True
        if loadgreyscale:
            full_size_image = cv2.imread(meta["path"], cv2.IMREAD_GRAYSCALE).astype(np.float32)
            full_size_image = np.transpose(np.stack([full_size_image,
                                                     full_size_image, full_size_image], axis=0), (1, 2, 0)).astype(
                np.uint8)
            full_size_image = np.ascontiguousarray(full_size_image)
        else:
            full_size_image = cv2.imread(meta["path"], cv2.IMREAD_COLOR).astype(np.uint8)
            full_size_image = cv2.cvtColor(full_size_image, cv2.COLOR_BGR2RGB)

        img = img.to(config.device)
        boxes, contours, scores = predict_image(fitter, img, threshold=config.conf_threshold,
                                                orig_shape=(meta["height"], meta["width"]))

        # compute growth metrics
        eccentricity_list = contours_to_eccentricity(contours)
        size_list = get_pixel_area_in_percent_from_contours(contours, full_size_image.shape[0:2])
        darkness_list = 1 - np.array(brightness_by_contours(contours, full_size_image))

        mean_eccentricity = np.mean(eccentricity_list) if len(eccentricity_list) != 0 else 0
        median_size = np.median(size_list) if len(size_list) != 0 else 0
        mean_darkness = np.mean(darkness_list) if len(darkness_list) != 0 else 0

        std_eccentricity = np.std(eccentricity_list) if len(eccentricity_list) != 0 else 0
        std_size = np.std(size_list) if len(size_list) != 0 else 0
        std_darkness = np.std(darkness_list) if len(darkness_list) != 0 else 0

        total_area = np.sum(size_list)

        if config.generate_plots:
            for iii, (box, contour, score) in enumerate(zip(boxes, contours, scores)):
                try:
                    cv2.drawContours(full_size_image, [contour], -1, (0, 0, 255), thickness=5)
                except:
                    print("Error")

            outp = OUTPUT_FOLDER + meta["path"].split("/")[-1]
            cv2.imwrite(outp, full_size_image)
            print("Predictions saved to ", outp)

        results.append([meta["path"], len(boxes), meta["height"], meta["width"],
                        mean_eccentricity, median_size, mean_darkness,
                        std_eccentricity, std_size, std_darkness, total_area])

    result = pd.DataFrame(results,
                          columns=["path", "count", "height", "width", "ecc_mean", "size_median", "darkness_mean",
                                   "ecc_std", "size_std", "darkness_std", "total_area"])

    result.to_csv(OUTPUT_FOLDER + "/inference_result.csv", index=False)
    print("Saved resultfile to ", OUTPUT_FOLDER)


if __name__ == '__main__':
    IMAGE_FOLDER = sys.argv[1]
    OUTPUT_FOLDER = sys.argv[2]
    inference(IMAGE_FOLDER, OUTPUT_FOLDER)