#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from goat.engine import FitterMaskRCNN
from goat.model import maskRCNNModel
from goat.dataset import MaskRCNNDataset

from config import Config

config = Config()


def check_dataframe_validity(df):
    unique_states = df["state"].unique()
    for state in unique_states:
        if state not in ["train", "val", "test", "threshold_selection"]:
            print("Dataframe error: state can only be ['train', 'val', 'test', 'threshold_selection']")

    if len(df["state"].unique()) == 1:

        if df["state"].unique()[0] == "train":
            print("Only found 'train' in state column.")

        if df["state"].unique()[0] == "val":
            print("Only found 'val' in state column.")

        if df["state"].unique()[0] == "test":
            print("Only found 'test' in state column.")
            if len(df) == 604:  # check if template dataset
                print("Overwriting 'test' with 'train'/'val' to be able to run training on examplary data.")
                unique_paths = df["path"].unique()
                train = unique_paths[0:6]
                val = unique_paths[6:]
                df.loc[df["path"].isin(train), "state"] = "train"
                df.loc[df["path"].isin(val), "state"] = "val"

        if len(df["state"].unique()) == "threshold_selection":
            print("Only found 'threshold_selection' in state column.")

    return df


def train(DATAFRAME):
    df = pd.read_csv(DATAFRAME)
    df = check_dataframe_validity(df)

    n_images = df["path"].unique().shape[0]
    gr = df.groupby("path", as_index=False).count()
    over_n = gr[gr["filename"] > config.restrict_nr]["path"]
    df = df[~df["path"].isin(over_n)]

    print(f"Dropping {len(over_n)} of {n_images} images with over {config.restrict_nr} organoids.")
    print(df.groupby(["state", "kind"]).nunique()["filename"])

    train_imgs = df[df.state == "train"].path.unique()
    train_boxes = [df[df.path == x][["x", "y", "width", "height"]].values for x in train_imgs]
    train_rle_strings = [df[df.path == x]["rle"].values for x in train_imgs]

    val_imgs = df[df.state == "val"].path.unique()
    val_boxes = [df[df.path == x][["x", "y", "width", "height"]].values for x in val_imgs]
    val_rle_strings = [df[df.path == x]["rle"].values for x in val_imgs]

    train_ds = MaskRCNNDataset(df, train_imgs, train_boxes, train_rle_strings, "train")
    val_ds = MaskRCNNDataset(df, val_imgs, val_boxes, val_rle_strings, "val")

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=config.num_workers, collate_fn=collate_fn)

    model = maskRCNNModel()
    fitter = FitterMaskRCNN(model, config.device, config)

    if config.continue_path is not None:
        print("Continuing:", config.continue_path)
        fitter.load(config.continue_path)

    fitter.fit(train_loader, val_loader)


if __name__ == '__main__':
    DATAFRAME = sys.argv[1]
    train(DATAFRAME)
