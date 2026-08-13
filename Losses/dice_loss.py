import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.Module):
    """
    Dice Loss for multi-class semantic segmentation.


    Input:

        prediction:
            Tensor[B,C,H,W]

            Output from network.


        target:

            Tensor[B,H,W]

            Ground truth mask.


    Dice Formula:

        Dice =

        2 * |Prediction ∩ GroundTruth|
        -------------------------------
        |Prediction| + |GroundTruth|


        Loss:

        1 - Dice

    """


    def __init__(
        self,
        smooth=1e-5,
        ignore_index=255
    ):

        super().__init__()


        self.smooth = smooth

        self.ignore_index = ignore_index



    def forward(
        self,
        prediction,
        target
    ):


        # --------------------------------
        # prediction:
        #
        # [B,C,H,W]
        #
        # softmax获得每个类别概率
        # --------------------------------

        prediction = torch.softmax(
            prediction,
            dim=1
        )


        num_classes = prediction.shape[1]


        # --------------------------------
        # 处理ignore区域
        # --------------------------------

        if self.ignore_index is not None:

            mask = (
                target != self.ignore_index
            )


            target = target.clone()


            target[
                ~mask
            ] = 0



        # --------------------------------
        # one-hot编码
        #
        # target:
        #
        # [B,H,W]
        #
        # ->
        #
        # [B,C,H,W]
        #
        # --------------------------------


        target_one_hot = F.one_hot(
            target,
            num_classes=num_classes
        )


        target_one_hot = target_one_hot.permute(
            0,
            3,
            1,
            2
        ).float()



        # --------------------------------
        # Dice计算
        # --------------------------------


        dims = (
            0,
            2,
            3
        )


        intersection = torch.sum(
            prediction * target_one_hot,
            dims
        )


        union = torch.sum(
            prediction,
            dims
        ) + torch.sum(
            target_one_hot,
            dims
        )


        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            union + self.smooth
        )


        dice_loss = 1 - dice_score.mean()



        return dice_loss