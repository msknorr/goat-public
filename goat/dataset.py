import numpy as np
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2
import torch
from config import Config
from goat.utils import rle_decode


config = Config()

def get_train_transforms():
    return A.Compose(
        [
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.20, sat_shift_limit=0.10,
                                     val_shift_limit=0.20, p=1),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=1),
            ], p=0.7),

            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.30, scale_limit=[-0.5, 1.0], rotate_limit=45, interpolation=1, # weights3/ -0.5 - 1.0
                                   border_mode=0, always_apply=False, p=1),
                A.RandomResizedCrop(512, 512, scale=(0.1, 1.0), ratio=(0.75, 1.33), interpolation=1,
                                    always_apply=False, p=1),
            ], p=0.5),
            A.ToGray(p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Blur(blur_limit=15, p=0.5),
            A.Resize(width=config.resize, height=config.resize, p=1),

        ],
        p=1.0,
        bbox_params={'format': 'pascal_voc', 'min_area': 2, 'min_visibility': 0, 'label_fields': ['category_id']}
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.ToGray(p=1),
            A.Resize(height=config.resize, width=config.resize, p=1.0),
        ],
        p=1.0,
        bbox_params={'format': 'pascal_voc', 'min_area': 2, 'min_visibility': 0, 'label_fields': ['category_id']}
    )


def get_inference_transforms():
    return A.Compose(
        [
            #A.ToGray(p=1),
            A.Resize(height=config.resize, width=config.resize, p=1.0),
        ],
        p=1.0,
    )


class InferenceMaskRCNNDataset(Dataset):
    def __init__(self, img_paths, resize):
        """
        Dataset rapper for inference images.
        :param img_paths: list of image paths
        :param resize: e.g. 512
        """
        super().__init__()
        self.img_paths = img_paths
        self.resize = resize
        self.transforms_val = get_inference_transforms()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        img = self.load_image(image_path)
        meta = {"height": img.shape[0], "width": img.shape[1], "path": image_path}
        data = {"image": img}
        augmented = self.transforms_val(**data)
        img = augmented['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img, meta

    def load_image(self, img_path):  # load greyscale at inference because webapp does it to save MBs
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image = np.transpose(np.stack([image, image, image], axis=0), (1, 2, 0))
        image /= 255.0
        return image


class MaskRCNNDataset(Dataset):
    def __init__(self, df, img_paths, gt_boxes, gt_rle_strings, datatype="train"):
        super().__init__()
        assert datatype in ["train", "val", "test", "inference", None]

        self.img_paths = img_paths
        self.gt_boxes = gt_boxes
        self.gt_rle_strings = gt_rle_strings
        assert (len(gt_boxes) == len(gt_rle_strings))
        self.mode = datatype


        if datatype == "train":
            self.transforms = get_train_transforms()
        else:
            self.transforms = get_valid_transforms()

        self.rles = []
        self.df = df

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image_path = self.img_paths[index]

        img = self.load_image(image_path)
        boxes = self.load_boxes(index)
        masks = self.load_masks(index, img.shape[0:2])
        labels = torch.ones((boxes.shape[0]), dtype=torch.int64)
        masks = [*masks]

        while True:
            data = {"image": img, "bboxes": boxes, "masks": masks, "category_id": np.arange(len(boxes))}
            augmented = self.transforms(**data)

            aug_img = augmented['image']
            aug_masks = augmented['masks']
            aug_boxes = augmented['bboxes']

            if len(aug_boxes) > 0:
                break

        img = np.array(aug_img)
        boxes = np.array(aug_boxes)

        # unlinke masks and labels, labelfields are dropped when boxes are dropped
        masks = np.array(aug_masks)[augmented["category_id"]]
        labels = np.array(labels)[augmented["category_id"]]

        new_boxes = []
        for mask, box in zip(masks, boxes):
            if np.sum(mask) > 4:
                y, x = np.where(mask)
                new_boxes.append([max(0, min(x)-1),
                                  max(0, min(y)-1),
                                  min(max(x)+1, mask.shape[0]), # todo check here
                                  min(max(y)+1, mask.shape[1])])
            else:
                new_boxes.append(box)
        boxes = np.array(new_boxes)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([index])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes)), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
                  "iscrowd": iscrowd}

        img = torch.from_numpy(img).permute(2, 0, 1).float()

        return img.float(), target, {"path": image_path, "kind": self.df[self.df.path == image_path]["kind"].values[0]}

    def load_image(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        return image

    def load_boxes(self, index):
        boxess = self.gt_boxes[index].copy()
        assert len(boxess) > 0
        boxess[:, 2] -= 1
        boxess[:, 3] -= 1
        # convert from coco (x,y,w,h) to pascal_voc (x1,y1,x2,y2)
        boxess[:, 2] = boxess[:, 0] + boxess[:, 2]
        boxess[:, 3] = boxess[:, 1] + boxess[:, 3]
        return boxess

    def load_masks(self, index, shape):
        rles = self.gt_rle_strings[index]
        binary = np.zeros((len(rles), *shape), dtype=np.int8)

        for i, rle in enumerate(rles):
            if type(rle) != float:
                binary[i] = rle_decode(rle, shape)
        return binary