import albumentations as A

from albumentations.pytorch import ToTensorV2



def get_train_transform(
    image_size=512
):
    """
    Training augmentation.

    Image and mask
    are transformed together.
    """


    transform = A.Compose(
        [

            A.Resize(
                image_size,
                image_size
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
    """


    transform = A.Compose(
        [

            A.Resize(
                image_size,
                image_size
            ),


            A.Normalize(
                mean=(0.485,0.456,0.406),
                std=(0.229,0.224,0.225)
            ),


            ToTensorV2()

        ]
    )


    return transform