import torch
import torch.nn.functional as F



def dice_score(
    prediction,
    target,
    num_classes,
    ignore_index=255,
    smooth=1e-5
):
    """
    Calculate Dice Score.


    Dice:

        2 * intersection
        -----------------
        prediction + target



    Args:


        prediction:

            Tensor[B,C,H,W]


        target:

            Tensor[B,H,W]


    Returns:

        float

    """


    prediction = torch.argmax(
        prediction,
        dim=1
    )



    if ignore_index is not None:

        mask = (
            target != ignore_index
        )


        prediction = prediction[mask]

        target = target[mask]



    dice_list = []



    for cls in range(num_classes):


        pred_cls = (
            prediction == cls
        ).float()


        target_cls = (
            target == cls
        ).float()



        intersection = torch.sum(
            pred_cls * target_cls
        )



        denominator = (
            torch.sum(pred_cls)
            +
            torch.sum(target_cls)
        )



        if denominator == 0:

            continue



        dice = (

            2 * intersection + smooth

        ) / (

            denominator + smooth

        )



        dice_list.append(
            dice
        )



    if len(dice_list)==0:

        return 0.0



    dice = torch.mean(
        torch.stack(dice_list)
    )


    return dice.item()