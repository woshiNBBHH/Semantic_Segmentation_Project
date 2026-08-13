import albumentations as A

from albumentations.pytorch import ToTensorV2



def get_train_transform(
    image_size=512
):
    """
    Training augmentation.

    Image and mask
    are transformed together.

    Args:
        image_size: int (e.g. 512) or tuple/list (H, W) e.g. (512, 512)
    """

    if isinstance(image_size, (tuple, list)):
        height, width = image_size
    else:
        height = width = image_size


    transform = A.Compose(
        [

            A.Resize(
                height,
                width
            ),


            A.HorizontalFlip(
                p=0.5
            ),


            A.RandomBrightnessContrast(
                p=0.2
            ),


            A.Normalize(
                mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)
            ),


            ToTensorV2()

        ]
    )


    return transform




def get_val_transform(
    image_size=512
):
    """
    Validation/Test transform.

    Only preprocessing.

    Args:
        image_size: int (e.g. 512) or tuple/list (H, W) e.g. (512, 512)
    """

    if isinstance(image_size, (tuple, list)):
        height, width = image_size
    else:
        height = width = image_size


    transform = A.Compose(
        [

            A.Resize(
                height,
                width
            ),


            A.Normalize(
                mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)
            ),


            ToTensorV2()

        ]
    )


    return transform