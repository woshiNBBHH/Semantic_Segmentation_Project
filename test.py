"""
Test script for semantic segmentation models.

Usage:
    python test.py --config Config.fcn_baseline --split val
    python test.py --config Config.fcn_baseline --split test --checkpoint Experiments/fcn_baseline/best.pth
"""

import os
import argparse
import importlib

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader


# ==========================
# Parse arguments
# ==========================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--config",
    type=str,
    default="Config.fcn_baseline",
    help="Python module path of the config file"
)

parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to model checkpoint. If None, uses Experiments/{EXP_NAME}/best.pth"
)

parser.add_argument(
    "--split",
    type=str,
    default="val",
    choices=["train", "val", "test"],
    help="Dataset split to evaluate"
)

parser.add_argument(
    "--num_vis",
    type=int,
    default=5,
    help="Number of prediction images to save"
)

args = parser.parse_args()


# ==========================
# Dynamic config import
# ==========================

Config = importlib.import_module(args.config).Config


from Datasets.dataset import VOCDataset
from Datasets.transforms import get_val_transform
from Models.Segmentation.fcn import FCN
from Metrics import pixel_accuracy, mean_iou, dice_score



# ==========================
# Main
# ==========================

@torch.no_grad()
def main():

    device = torch.device(
        Config.DEVICE
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")


    # --------------------------
    # Dataset
    # --------------------------

    dataset = VOCDataset(
        root=Config.DATA_ROOT,
        split=args.split,
        transform=get_val_transform(Config.IMAGE_SIZE)
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )


    # --------------------------
    # Model
    # --------------------------

    model = FCN(
        num_classes=Config.NUM_CLASSES,
        pretrained=False
    )

    model.to(device)


    # --------------------------
    # Load checkpoint
    # --------------------------

    checkpoint_path = args.checkpoint

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            "Experiments",
            Config.EXP_NAME,
            "best.pth"
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)

    print(f"Loaded checkpoint: {checkpoint_path}")


    # --------------------------
    # Evaluation
    # --------------------------

    model.eval()

    total_acc = 0.0
    total_miou = 0.0
    total_dice = 0.0

    saved = 0

    save_dir = os.path.join(
        "Experiments",
        Config.EXP_NAME,
        "pred_images"
    )

    os.makedirs(save_dir, exist_ok=True)


    for batch_idx, (images, masks) in enumerate(loader):

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        acc = pixel_accuracy(outputs, masks)
        miou = mean_iou(outputs, masks, Config.NUM_CLASSES)
        dice = dice_score(outputs, masks, Config.NUM_CLASSES)

        total_acc += acc
        total_miou += miou
        total_dice += dice


        # Save visualizations
        if saved < args.num_vis:

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            images_np = images.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(min(args.num_vis - saved, images.size(0))):

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                img = images_np[i].transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225])
                img = img + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                axes[0].imshow(img)
                axes[0].set_title("Image")
                axes[0].axis("off")

                axes[1].imshow(masks_np[i])
                axes[1].set_title("Ground Truth")
                axes[1].axis("off")

                axes[2].imshow(preds[i])
                axes[2].set_title("Prediction")
                axes[2].axis("off")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        save_dir,
                        f"pred_{saved}.png"
                    ),
                    dpi=150,
                    bbox_inches="tight"
                )
                plt.close(fig)

                saved += 1

                if saved >= args.num_vis:
                    break


    num_batches = len(loader)

    avg_acc = total_acc / num_batches
    avg_miou = total_miou / num_batches
    avg_dice = total_dice / num_batches


    print("=" * 50)
    print(f"Split: {args.split}")
    print(f"Samples: {len(dataset)}")
    print(f"Pixel Accuracy: {avg_acc:.4f}")
    print(f"mIoU: {avg_miou:.4f}")
    print(f"Dice Score: {avg_dice:.4f}")
    print("=" * 50)


    # Save metrics
    metrics_path = os.path.join(
        save_dir,
        "test_metrics.txt"
    )

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Split: {args.split}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Pixel Accuracy: {avg_acc:.4f}\n")
        f.write(f"mIoU: {avg_miou:.4f}\n")
        f.write(f"Dice Score: {avg_dice:.4f}\n")

    print(f"Metrics saved to: {metrics_path}")
    print(f"Visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()
