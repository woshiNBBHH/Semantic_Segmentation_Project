import os
import cv2
import torch

from torch.utils.data import Dataset


class VOCDataset(Dataset):
    """
    Pascal VOC 2012 semantic segmentation dataset.
    VOC structure:
    VOC2012/
        JPEGImages/
        SegmentationClass/
        ImageSets/
            Segmentation/
                train.txt
                val.txt
    Return:
        image:
            Tensor[C,H,W]
        mask:
            Tensor[H,W]
    """

    def __init__(self,root,split="train", transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(
            root,
            "JPEGImages"
        )
        self.mask_dir = os.path.join(
            root,
            "SegmentationClass"
        )
        # 读取官方划分文件
        split_file = os.path.join(
            root,
            "ImageSets",
            "Segmentation",
            split + ".txt"
        )

        with open(
            split_file,
            "r"
        ) as f:
            self.images = [
                line.strip()
                for line in f.readlines()
            ]

    def __len__(self):
        return len(
            self.images
        )

    def __getitem__(
        self,
        index
    ):
        name = self.images[index]
        image_path = os.path.join(
            self.image_dir,
            name + ".jpg"
        )

        mask_path = os.path.join(
            self.mask_dir,
            name + ".png"
        )

        # =========================
        # read image
        # =========================
        image = cv2.imread(
            image_path
        )
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
        # =========================
        # read mask
        # =========================
        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )
        # =========================
        # transform
        # =========================
        if self.transform:
            transformed = self.transform(
                image=image,
                mask=mask
            )
            image = transformed["image"]
            mask = transformed["mask"]

        else:
            image = torch.from_numpy(
                image
            ).permute(
                2,0,1
            ).float()

            mask = torch.from_numpy(
                mask
            )

        mask = mask.long()

        return image, mask