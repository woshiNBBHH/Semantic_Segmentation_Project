import torch



def mean_iou(
    prediction,
    target,
    num_classes,
    ignore_index=255
):
    """
    Calculate Mean Intersection over Union.


    IoU:

        Intersection
        ------------
        Union


    Args:


        prediction:

            Tensor[B,C,H,W]


        target:

            Tensor[B,H,W]


        num_classes:

            Number of classes



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



    ious = []



    for cls in range(num_classes):


        pred_cls = (
            prediction == cls
        )


        target_cls = (
            target == cls
        )



        intersection = torch.sum(
            pred_cls & target_cls
        )



        union = torch.sum(
            pred_cls | target_cls
        )



        # 如果该类别不存在

        if union == 0:

            continue



        iou = (
            intersection.float()
            /
            union.float()
        )


        ious.append(iou)



    if len(ious)==0:

        return 0.0



    miou = torch.mean(
        torch.stack(ious)
    )


    return miou.item()