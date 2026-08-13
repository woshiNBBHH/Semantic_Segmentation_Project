import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """
    Multi-class Cross Entropy Loss for semantic segmentation.


    Input:
        prediction:
            Tensor[B, C, H, W]

        target:
            Tensor[B, H, W]


    Example:

        pred = model(image)

        loss = criterion(pred, mask)

    """


    def __init__(
        self,
        weight=None,
        ignore_index=255
    ):
        """
        Args:

            weight:
                Class weight.
                Used for class imbalance.

                Shape:
                    [num_classes]


            ignore_index:
                Pixels with this label
                will not contribute to loss.

                Default:
                    255
        """

        super().__init__()


        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index
        )


    def forward(
        self,
        prediction,
        target
    ):

        loss = self.loss(
            prediction,
            target
        )


        return loss