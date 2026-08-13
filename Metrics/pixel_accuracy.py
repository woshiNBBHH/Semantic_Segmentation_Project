import torch


def pixel_accuracy(
        prediction,
        target,
        ignore_index=255
):
    """
    Calculate Pixel Accuracy.


    Formula:

        Pixel Accuracy =

        Number of correctly classified pixels
        ------------------------------------
        Total valid pixels


    Args:

        prediction:

            Model output

            Tensor:
            [B,C,H,W]


        target:

            Ground truth mask

            Tensor:
            [B,H,W]


    Returns:

        float
    """

    # prediction:
    # [B,C,H,W]
    #
    # -> predicted class

    prediction = torch.argmax(
        prediction,
        dim=1
    )

    # remove ignore pixels

    if ignore_index is not None:
        mask = (
                target != ignore_index
        )

        prediction = prediction[mask]

        target = target[mask]

    correct = torch.sum(
        prediction == target
    )

    total = target.numel()

    if total == 0:
        return 0.0

    acc = (
            correct.float()
            /
            total
    )

    return acc.item()