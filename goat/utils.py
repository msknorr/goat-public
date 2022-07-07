import cv2
import numpy as np
from config import Config


config = Config()


def brightness_by_contours(contours, img_fullsize):
    brightnesses = []
    for cont in contours:
        mask = np.zeros(img_fullsize.shape[0:2], np.uint8)
        cont = np.array(cont).astype(np.int32)
        cv2.fillConvexPoly(mask, cont, 255)
        mean = cv2.mean(img_fullsize, mask=mask)
        assert mean[0] == mean[1]  # fixed by using cv2.IMREAD_GRAYSCALE
        brightnesses.append(mean[0] / 255)
    return brightnesses


def get_pixel_area_in_percent_from_contours(contours, rects):
    areas = []
    allpixels = rects[0] * rects[1]
    for cont in contours:
        cont = np.array(cont).astype(np.int32)
        area = cv2.contourArea(cont) / allpixels
        areas.append(area)
    return areas


def contours_to_eccentricity(contours):
    eccs = []

    for cont in contours:
        if len(cont) < 5:  # probably redundant
            eccs.append(np.nan)
            continue
        cont = np.array(cont).astype(np.int32)
        ellipse = cv2.fitEllipse(cont)
        longax, shortax = max(ellipse[1]), min(ellipse[1])
        ecc = np.sqrt(1 - (shortax ** 2 / longax ** 2))
        eccs.append(ecc)
    return eccs


def get_contours_from_prediction_masks(masks, rects):
    contour_list = []
    id_to_drop = []
    for i, mask in enumerate(masks):
        mask = mask.squeeze()
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            id_to_drop.append(i)
            contour_list.append(np.nan)
            continue
        cont = max(contours, key=cv2.contourArea)
        # upscale to original
        coef_y = rects[0] / config.resize
        coef_x = rects[1] / config.resize
        cont[:, :, 0] = cont[:, :, 0] * coef_x
        cont[:, :, 1] = cont[:, :, 1] * coef_y
        nr_points = len(cont)
        if nr_points < 5:  # eccentricity needs atleast 5 points
            id_to_drop.append(i)
            contour_list.append(np.nan)
            continue
        cont = np.array(cont)
        contour_list.append(cont)

    return np.array(contour_list, dtype=object), id_to_drop


def contours_to_boxes(contours):
    boxes = []
    for cont in contours:
        cont = np.array(cont).astype(np.int32)
        x, y, w, h = cv2.boundingRect(cont)
        boxes.append([x, y, x + w, y + h])
    return boxes


def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


def read_image(path, grayscale=False):
    if grayscale:
        full_size_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        full_size_image = np.transpose(np.stack([full_size_image,
                                                 full_size_image, full_size_image], axis=0), (1, 2, 0)).astype(np.uint8)
        full_size_image = np.ascontiguousarray(full_size_image)
    else:
        full_size_image = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.uint8)
        full_size_image = cv2.cvtColor(full_size_image, cv2.COLOR_BGR2RGB)
    return full_size_image
