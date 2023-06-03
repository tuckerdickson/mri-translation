# import required modules
import torch
import torch.nn as nn

# -------------------------------------------------------------------------------------- #
# Class:       Discriminator                                                             #
# Description: This class defines the patch-GAN architecture for the descriminator       #
#              that is used in our cycle-GAN. The job of the discriminator is to         #
#              classify inputs (images as real or fake).                                 #
# -------------------------------------------------------------------------------------- #
class Discriminator(nn.Module):
    # function:    init
    # description: initializes class attributes (defines layers in model)
    # inputs:      none
    # outputs:     none
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # the patch-GAN architecture uses several conv->instance norm->leaky relu blocks
        # here, we use five blocks, followed by one final convolution to reduce the channels back to 1
        
        # block 1
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=64, 
            kernel_size=4, 
            stride=2, 
            padding=1, 
            bias=True, 
            padding_mode='reflect'
        )
        self.norm1 = nn.InstanceNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2)
               
        # block 2
        self.conv2 = nn.Conv2d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=4, 
            stride=2, 
            padding=1, 
            bias=True, 
            padding_mode='reflect'
        )
        self.norm2 = nn.InstanceNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2)
               
        # block 3
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=4, 
            stride=2,
            padding=1, 
            bias=True,
            padding_mode='reflect'
        )
        self.norm3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.LeakyReLU(0.2)
               
        # block 4
        self.conv4 = nn.Conv2d(
            in_channels=256, 
            out_channels=512, 
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True, 
            padding_mode='reflect'
        )
        self.norm4 = nn.InstanceNorm2d(512)
        self.relu4 = nn.LeakyReLU(0.2)
        
        # block 5
        self.conv5 = nn.Conv2d(
            in_channels=512,
            out_channels=512, 
            kernel_size=4, 
            stride=1, 
            padding=1,
            bias=True,
            padding_mode='reflect'
        )
        self.norm5 = nn.InstanceNorm2d(512)
        self.relu5 = nn.LeakyReLU(0.2)
        
        # one last convolution to reduce the number of channels back to 1
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1)

    # function:    forward
    # description: takes an image as input and passes it forward through the network
    # inputs:      x - a real or fake T1 or T2 image
    # outputs:     a prediction of whether the input image is real or fake
    def forward(self, x):
        # block 1
        conv1out = self.conv1(x)
        norm1out = self.norm1(conv1out)
        relu1out = self.relu1(norm1out)
                
        # block 2
        conv2out = self.conv2(relu1out)
        norm2out = self.norm2(conv2out)
        relu2out = self.relu2(norm2out)
                
        # block 3
        conv3out = self.conv3(relu2out)
        norm3out = self.norm3(conv3out)
        relu3out = self.relu3(norm3out)
                
        # block 4
        conv4out = self.conv4(relu3out)
        norm4out = self.norm4(conv4out)
        relu4out = self.relu4(norm4out)
                
        # block 5
        conv5out = self.conv5(relu4out)
        norm5out = self.norm5(conv5out)
        relu5out = self.relu5(norm5out)
                
        # final convolution
        conv6out = self.conv6(relu5out)
                
        # return the sigmoid of the conv6 output so that it's between 0 and 1
        return torch.sigmoid(conv6out)