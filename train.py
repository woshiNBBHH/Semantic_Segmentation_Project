import os
import argparse
import random
import time


import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader


# ==========================
# Config
# ==========================

from Config.fcn_baseline import Config



# ==========================
# Dataset
# ==========================

from Datasets.dataset import VOCDataset

from Datasets.transforms import (
    get_train_transform,
    get_val_transform
)



# ==========================
# Model
# ==========================

from Models.Segmentation.fcn import FCN



# ==========================
# Loss
# ==========================

from Losses import (
    CrossEntropyLoss,
    DiceLoss
)



# ==========================
# Metrics
# ==========================

from Metrics import (
    pixel_accuracy,
    mean_iou,
    dice_score
)



# ==========================
# Seed
# ==========================


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)



# ==========================
# Experiment Directory
# ==========================


def create_experiment_dir():


    save_dir = os.path.join(
        "Experiments",
        Config.EXP_NAME
    )


    os.makedirs(
        save_dir,
        exist_ok=True
    )


    os.makedirs(
        os.path.join(
            save_dir,
            "pred_images"
        ),
        exist_ok=True
    )


    return save_dir



# ==========================
# Logger
# ==========================


def write_log(
    save_dir,
    message
):

    print(message)


    with open(
        os.path.join(
            save_dir,
            "log.txt"
        ),
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            message+"\n"
        )



# ==========================
# Save checkpoint
# ==========================


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    best_miou
):


    state = {

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "best_miou":
            best_miou

    }


    torch.save(
        state,
        path
    )



# ==========================
# Train one epoch
# ==========================


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):


    model.train()


    total_loss = 0



    for images, masks in loader:


        images = images.to(device)

        masks = masks.to(device)



        optimizer.zero_grad()



        outputs = model(images)



        loss = criterion(
            outputs,
            masks
        )



        loss.backward()


        optimizer.step()



        total_loss += loss.item()



    return total_loss / len(loader)



# ==========================
# Validation
# ==========================


@torch.no_grad()

def validate(
    model,
    loader,
    device
):


    model.eval()



    total_acc = 0

    total_miou = 0

    total_dice = 0



    for images,masks in loader:


        images = images.to(device)

        masks = masks.to(device)



        outputs = model(images)



        acc = pixel_accuracy(
            outputs,
            masks
        )


        miou = mean_iou(
            outputs,
            masks,
            Config.NUM_CLASSES
        )


        dice = dice_score(
            outputs,
            masks,
            Config.NUM_CLASSES
        )


        total_acc += acc

        total_miou += miou

        total_dice += dice



    length = len(loader)



    return (

        total_acc / length,

        total_miou / length,

        total_dice / length

    )



# ==========================
# Main
# ==========================


def main(resume=None):


    set_seed(
        Config.SEED
    )


    save_dir = create_experiment_dir()



    device = torch.device(
        Config.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )



    write_log(
        save_dir,
        f"Device: {device}"
    )



    # --------------------------
    # Dataset
    # --------------------------

    train_dataset = VOCDataset(

        root=Config.DATA_ROOT,

        split="train",

        transform=get_train_transform(
            Config.IMAGE_SIZE
        )

    )

    val_dataset = VOCDataset(

        root=Config.DATA_ROOT,

        split="val",

        transform=get_val_transform(
            Config.IMAGE_SIZE
        )

    )



    train_loader = DataLoader(

        train_dataset,

        batch_size=Config.BATCH_SIZE,

        shuffle=True,

        num_workers=Config.NUM_WORKERS

    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=Config.BATCH_SIZE,

        shuffle=False,

        num_workers=Config.NUM_WORKERS

    )



    # --------------------------
    # Model
    # --------------------------


    model = FCN(

        num_classes=Config.NUM_CLASSES,

        pretrained=Config.PRETRAINED

    )


    model.to(device)



    # --------------------------
    # Loss
    # --------------------------


    ce_loss = CrossEntropyLoss(
        ignore_index=Config.IGNORE_INDEX
    )


    dice_loss = DiceLoss(
        ignore_index=Config.IGNORE_INDEX
    )



    def criterion(pred,mask):


        loss = ce_loss(
            pred,
            mask
        )


        if Config.USE_DICE_LOSS:


            loss += (
                Config.DICE_WEIGHT
                *
                dice_loss(
                    pred,
                    mask
                )
            )


        return loss



    # --------------------------
    # Optimizer
    # --------------------------


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=Config.LR,

        weight_decay=Config.WEIGHT_DECAY

    )



    start_epoch = 0

    best_miou = 0



    # --------------------------
    # Resume
    # --------------------------


    if resume:


        checkpoint = torch.load(
            resume,
            map_location=device
        )


        model.load_state_dict(
            checkpoint["model_state_dict"]
        )


        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )


        start_epoch = (
            checkpoint["epoch"]
            +
            1
        )


        best_miou = checkpoint["best_miou"]



        write_log(
            save_dir,
            f"Resume from epoch {start_epoch}"
        )



    # --------------------------
    # Training
    # --------------------------


    for epoch in range(
        start_epoch,
        Config.EPOCHS
    ):


        start=time.time()



        train_loss = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            device

        )



        acc,miou,dice = validate(

            model,

            val_loader,

            device

        )



        message = (

            f"Epoch [{epoch+1}/{Config.EPOCHS}] "

            f"Loss:{train_loss:.4f} "

            f"Acc:{acc:.4f} "

            f"mIoU:{miou:.4f} "

            f"Dice:{dice:.4f} "

            f"Time:{time.time()-start:.1f}s"

        )


        write_log(
            save_dir,
            message
        )



        # latest

        save_checkpoint(

            os.path.join(
                save_dir,
                "latest.pth"
            ),

            epoch,

            model,

            optimizer,

            best_miou

        )



        # best

        if miou > best_miou:


            best_miou = miou


            torch.save(

                model.state_dict(),

                os.path.join(
                    save_dir,
                    "best.pth"
                )

            )



if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--resume",
        type=str,
        default=None
    )


    args = parser.parse_args()



    main(
        args.resume
    )