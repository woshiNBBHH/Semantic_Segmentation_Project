import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGG16Backbone(nn.Module):
    """
    VGG16 backbone for FCN.

    Return:

        x3:
            feature map after conv3

        x4:
            feature map after conv4

        x5:
            feature map after conv5

    Shape example:

        Input:

            [B,3,512,512]

        x3:

            [B,256,64,64]

        x4:

            [B,512,32,32]

        x5:

            [B,512,16,16]

    """

    def __init__(
        self,
        pretrained=True
    ):

        super().__init__()

        if pretrained:

            weights = VGG16_Weights.DEFAULT

        else:

            weights = None

        vgg = vgg16(
            weights=weights
        )

        features = vgg.features


        # VGG16结构:

        # 0-4:
        # block1

        # 5-9:
        # block2

        # 10-16:
        # block3

        # 17-23:
        # block4

        # 24-30:
        # block5



        self.block1 = nn.Sequential(
            *features[:5]
        )


        self.block2 = nn.Sequential(
            *features[5:10]
        )


        self.block3 = nn.Sequential(
            *features[10:17]
        )


        self.block4 = nn.Sequential(
            *features[17:24]
        )


        self.block5 = nn.Sequential(
            *features[24:31]
        )



    def forward(self,x):


        x = self.block1(x)


        x = self.block2(x)


        x3 = self.block3(x)


        x4 = self.block4(x3)


        x5 = self.block5(x4)



        return x3,x4,x5