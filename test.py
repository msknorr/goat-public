import sys
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os

from goat.engine import FitterMaskRCNN
from goat.model import maskRCNNModel, MaskRCNNPrediction, unscale_preds
from goat.dataset import MaskRCNNDataset
from goat.utils import get_contours_from_prediction_masks, read_image
from config import Config

config = Config()

HIT_COLOR = (255,0,0)
FN_COLOR = (255,255,0)
FP_COLOR = (0,0,255)
BEST_CONF_THRESH = 0.9


def generate_test_results(fitter, test_ds):
    result = []
    for i, (img, target, meta) in enumerate(test_ds):
        print("Processing", meta["path"])
        full_size_image = read_image(meta["path"])
        h, w = full_size_image.shape[0:2]

        goat_prediction = model_prediction(fitter.model, img.unsqueeze(0), [target], meta, plot=False)
        best_confthresh = find_best_f1(goat_prediction)

        goat_gt_matches = goat_prediction.compute_matches(0.5, best_confthresh)[0]
        goat_pred_matches = goat_prediction.compute_matches(0.5, best_confthresh)[1]

        pred_masks = goat_prediction.pred_masks[goat_prediction.pred_scores >= best_confthresh]
        pred_boxes = goat_prediction.pred_boxes[goat_prediction.pred_scores >= best_confthresh]
        pred_boxes, pred_masks = unscale_preds(pred_boxes, pred_masks, w, h)

        for box, mask, hit in zip(pred_boxes, pred_masks, goat_pred_matches):
            mask = ((mask) > 0.5).astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            color = HIT_COLOR if hit != -1 else FP_COLOR
            cv2.drawContours(full_size_image, contours, -1, color, thickness=5)

        boxes_gt, masks_gt = unscale_preds(goat_prediction.gt_boxes, goat_prediction.gt_masks, w, h)
        for mask, hit in zip(masks_gt, goat_gt_matches):
            mask = ((mask) > 0.5).astype(np.uint8)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if hit == -1:
                cv2.drawContours(full_size_image, contours, -1, FN_COLOR, thickness=5)

        metrics = goat_prediction.compute_metrics(iou_threshold=0.5, confidence_threshold=BEST_CONF_THRESH)
        f1 = metrics["f1"]
        pre = metrics["precision"]
        rec = metrics["recall"]
        ap50 = goat_prediction.compute_scores(0.5)[0]
        ap75 = goat_prediction.compute_scores(0.75)[0]
        ap5095 = goat_prediction.compute_ap_range()
        result.append([meta["path"], goat_prediction.organoid, f1, pre, rec, ap50, ap75, ap5095])

        fig, ax = plt.subplots()
        plt.imshow(full_size_image)
        plt.axis("off")
        plt.title(f"AP-50 {round(ap50, 2)}, Thr {best_confthresh}")
        outp = OUTPUT_FOLDER + meta["path"].split("/")[-1]
        plt.savefig(outp, dpi=256, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    result = pd.DataFrame(result, columns=["path", "kind", "f1", "precision", "recall", "ap50", "ap75", "ap5095"])
    return result


def model_prediction(model, img, target, meta, plot=False):
    model.eval()
    organoid_type = meta["kind"]
    img = torch.stack([*img]).to(config.device)
    with torch.no_grad():
        pred = model(img)
    pred_boxes = pred[0]["boxes"].detach().cpu().numpy()
    pred_scores = pred[0]["scores"].detach().cpu().numpy()
    pred_masks = (pred[0]["masks"].detach().cpu().numpy()[:, 0, :, :] > 0.5).astype(np.uint8)  # DAS HIER IST NEU
    pred_labels = pred[0]["labels"].detach().cpu().numpy()
    
    gt_boxes = target[0]["boxes"].detach().cpu().numpy()
    gt_masks = target[0]["masks"].detach().cpu().numpy()
    gt_labels = target[0]["labels"].detach().cpu().numpy()

    if plot:
        fig, ax = plt.subplots(ncols=5, figsize=(12, 4))
        ax[0].imshow(img[0].permute(1,2,0).detach().cpu().numpy())
        ax[1].imshow(np.mean(gt_masks, axis=0))
        ax[2].imshow(np.mean(pred_masks, axis=0))

        t_gt = np.zeros((512, 512)).astype(np.uint8)
        for box in gt_boxes:
            cv2.rectangle(t_gt, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (255, 0, 0), 2)
        ax[3].imshow(t_gt)

        t_pe = np.zeros((512, 512)).astype(np.uint8)
        for box in pred_boxes:
            cv2.rectangle(t_pe, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (255, 0, 0), 2)
        ax[4].imshow(t_pe)
        plt.show()
    
    return MaskRCNNPrediction(organoid_type, gt_boxes, gt_labels, gt_masks,
                              pred_boxes, pred_labels, pred_scores, pred_masks, meta['path'])


def find_best_f1(mrcnn_prediction):
    scores_ioter=[]
    for ctf in np.arange(0,1.05,0.05):
        scores_ioter.append(mrcnn_prediction.compute_metrics(iou_threshold=0.5, confidence_threshold=ctf)["f1"])
    # best_f1_goat = round(np.max(scores_ioter), 2)
    return round(np.arange(0,1.05,0.05)[np.argmax(scores_ioter)], 2)


def test(DATAFRAME, OUTPUT_FOLDER):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    df = pd.read_csv(DATAFRAME)
    print("Found dataframe with", df.filename.unique().shape[0], "images")
    test_imgs = df[df.state == "test"].path.unique()
    test_boxes = [df[df.path == x][["x", "y", "width", "height"]].values for x in test_imgs]
    test_rle_strings = [df[df.path == x]["rle"].values for x in test_imgs]

    test_ds = MaskRCNNDataset(df, test_imgs, test_boxes, test_rle_strings, "val")

    model = maskRCNNModel()
    fitter = FitterMaskRCNN(model, config.device, config)
    fitter.load(config.model_weights)

    result = generate_test_results(fitter, test_ds)
    result.to_csv(OUTPUT_FOLDER + "test_result.csv", index=False)
    print("Saved result file to ", OUTPUT_FOLDER)


if __name__ == '__main__':
    DATAFRAME = sys.argv[1]
    OUTPUT_FOLDER = sys.argv[2]
    test(DATAFRAME, OUTPUT_FOLDER)