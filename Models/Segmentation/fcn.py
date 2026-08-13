import torch
import torch.nn as nn
import torch.nn.functional as F


from Models.Backbone.vgg16 import VGG16Backbone




class FCN(nn.Module):
    """
    Fully Convolutional Network.


    Backbone:

        VGG16


    Method:

        FCN-8s


    Input:

        [B,3,H,W]


    Output:

        [B,num_classes,H,W]

    """



    def __init__(
        self,
        num_classes,
        pretrained=True
    ):

        super().__init__()



        self.backbone = VGG16Backbone(
            pretrained=pretrained
        )



        # VGG16 conv5 feature

        self.score5 = nn.Conv2d(
            512,
            num_classes,
            kernel_size=1
        )


        # conv4 feature

        self.score4 = nn.Conv2d(
            512,
            num_classes,
            kernel_size=1
        )


        # conv3 feature

        self.score3 = nn.Conv2d(
            256,
            num_classes,
            kernel_size=1
        )



    def forward(self,x):


        input_size = x.shape[-2:]



        x3,x4,x5 = self.backbone(x)



        score5 = self.score5(x5)


        score4 = self.score4(x4)


        score3 = self.score3(x3)



        # 上采样到相同尺寸


        score5 = F.interpolate(
            score5,
            size=x4.shape[-2:],
            mode="bilinear",
            align_corners=False
        )


        fuse4 = score5 + score4



        fuse4 = F.interpolate(
            fuse4,
            size=x3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )


        fuse3 = fuse4 + score3



        output = F.interpolate(
            fuse3,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )


        return output