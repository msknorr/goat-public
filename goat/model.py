import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision
import numpy as np
import cv2
import goat.mrcnn_utils as mrcnn_utils
from goat.utils import get_contours_from_prediction_masks, contours_to_boxes

def maskRCNNModel():
    backbone = resnet_fpn_backbone('resnet34', pretrained=True, trainable_layers=4)
    model = MaskRCNN(backbone,
                      num_classes=2,
                      box_detections_per_img=1000,
                      box_nms_thresh=0.4,
                     )
    model.roi_heads.batch_size_per_image = 256
    model.rpn.batch_size_per_image = 128
    for param in model.parameters():
        param.requires_grad = True
    return model


def thresholded_prediction(fitter, img, threshold):
    """
    Predict Image sample and drop by threshold
    """
    assert type(img) == torch.Tensor
    pred = fitter.model(img.unsqueeze(0))
    scores = pred[0]["scores"]
    slicee = scores > threshold
    scores = pred[0]["scores"][slicee].detach().cpu().numpy()
    boxes = pred[0]["boxes"][slicee].detach().cpu().numpy()
    masks = pred[0]["masks"][slicee].squeeze(1).detach().cpu().numpy()
    masks = (masks > 0.5).astype(np.uint8)
    return scores, boxes, masks


def unscale_preds(boxes, masks, w, h):
    """
    Scale predictions back to original image size.
    """
    width_multiplier = w / masks.shape[-1]
    height_multiplier = h / masks.shape[-1]

    for i in range(len(boxes)):
        boxes[i][0] *= width_multiplier
        boxes[i][1] *= height_multiplier
        boxes[i][2] *= width_multiplier
        boxes[i][3] *= height_multiplier
        boxes[i] = [int(x) for x in boxes[i]]
    masks = np.array([cv2.resize(x, (w, h)) for x in masks])
    return boxes, masks


def predict_image(fitter, img, threshold, orig_shape):
    """
    Predict image and cutoff at threshold.
    :param fitter:
    :param img: [3, x, y]
    :param threshold: model confidence threshold
    :param orig_shape: tuple(height, idth)
    :return: boxes, masks
    """
    assert len(orig_shape) == 2  # x, y
    h, w = orig_shape

    fitter.model.eval()
    scores, boxes, masks = thresholded_prediction(fitter, img, threshold)

    contours, id_to_drop = get_contours_from_prediction_masks(masks, (h,w))

    contours = np.delete(contours, id_to_drop, axis=0)
    scores = np.delete(scores, id_to_drop, axis=0)
    assert len(scores) == len(contours)

    boxes = contours_to_boxes(contours)

    return boxes, contours, scores


class MaskRCNNPrediction:
    def __init__(self, organoid, gt_boxes, gt_labels, gt_masks,
                 pred_boxes, pred_labels, pred_scores, pred_masks, path):

        self.organoid = organoid  # kind
        self.gt_boxes = gt_boxes
        self.gt_labels = gt_labels
        self.gt_masks = gt_masks
        self.pred_boxes = pred_boxes
        self.pred_labels = pred_labels
        self.pred_scores = pred_scores
        self.pred_masks = pred_masks
        self.path = path
        # self.conf_thresh = conf_thresh

    def compute_ap_range(self):
        """
        Compute mAP
        """
        aps = []
        for iou_thresh in np.arange(0.5, 1.0, 0.05):
            ap = self.compute_scores(iou_thresh)[0]
            aps.append(ap)
        return np.mean(aps)

    def compute_scores(self, iou_threshold=0.5):
        """
        Compute confidence threshold indipended scores.
        """
        ap, precisions, recalls, overlaps = mrcnn_utils.compute_ap(
            gt_boxes=self.gt_boxes,
            gt_class_ids=self.gt_labels,
            gt_masks=np.transpose(self.gt_masks, (1, 2, 0)),
            pred_boxes=self.pred_boxes,
            pred_class_ids=self.pred_labels,
            pred_scores=self.pred_scores,
            pred_masks=np.transpose(self.pred_masks, (1, 2, 0)),
            iou_threshold=iou_threshold
        )
        return ap, precisions, recalls, overlaps

    def compute_metrics(self, iou_threshold=0.5, confidence_threshold=0.9):
        """
        Compute TP/FP/FN and F1 at confidence threshold.
        """
        pred_boxes = self.pred_boxes[self.pred_scores >= confidence_threshold]
        pred_masks = self.pred_masks[self.pred_scores >= confidence_threshold]
        pred_labels = self.pred_labels[self.pred_scores >= confidence_threshold]
        pred_scores = self.pred_scores[self.pred_scores >= confidence_threshold]
        pred_masks = np.transpose(pred_masks, (1, 2, 0))
        gt_masks = np.transpose(self.gt_masks, (1, 2, 0))

        gt_match, pred_match, overlaps = mrcnn_utils.compute_matches(
            gt_boxes=self.gt_boxes,
            gt_class_ids=self.gt_labels,
            gt_masks=gt_masks,
            pred_boxes=pred_boxes,
            pred_class_ids=pred_labels,
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            iou_threshold=iou_threshold,
        )

        TP = len(gt_match[gt_match != -1])
        FP = len(pred_match[pred_match == -1])
        FN = len(gt_match[gt_match == -1])

        if (TP + FP) != 0:
            precision = TP / (TP + FP)
        else:
            precision = 0
        if (TP + FN) != 0:
            recall = TP / (TP + FN)
        else:
            recall = 0
        if (precision + recall) != 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        metrics = {"f1": f1, "precision": precision, "recall": recall, "TP": TP, "FP": FP, "FN": FN}
        return metrics

    def compute_matches(self, iou_threshold=0.5, confidence_threshold=0.9):
        """
        Compute matches for plotting TP, TN, FP
        """
        pred_boxes = self.pred_boxes[self.pred_scores >= confidence_threshold]
        pred_masks = self.pred_masks[self.pred_scores >= confidence_threshold]
        pred_labels = self.pred_labels[self.pred_scores >= confidence_threshold]
        pred_scores = self.pred_scores[self.pred_scores >= confidence_threshold]
        pred_masks = np.transpose(pred_masks, (1, 2, 0))
        gt_masks = np.transpose(self.gt_masks, (1, 2, 0))

        gt_match, pred_match, overlaps = mrcnn_utils.compute_matches(
            gt_boxes=self.gt_boxes,
            gt_class_ids=self.gt_labels,
            gt_masks=gt_masks,
            pred_boxes=pred_boxes,
            pred_class_ids=pred_labels,
            pred_scores=pred_scores,
            pred_masks=pred_masks,
            iou_threshold=iou_threshold,
        )
        return gt_match, pred_match, overlaps

