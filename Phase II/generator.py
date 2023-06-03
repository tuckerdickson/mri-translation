# import required modules
import torch
import torch.nn as nn

# -------------------------------------------------------------------------------------- #
# Class:       DoubleConv2d                                                              #
# Description: This class defines the "double convolution", a building block in the      #
#              U-Net architecture. A double convolution consists of two back-to-back     #
#              convolution->batch norm->relu blocks.                                     #
# -------------------------------------------------------------------------------------- #
class DoubleConv2D(nn.Module):
    # function:    init
    # description: initializes class attributes (defines layers)
    # inputs:      inChannels - the number of channels of the input
    #              outChannels - the number of channels of the output
    # outputs:     none
    def __init__(self, inChannels, outChannels):
        super(DoubleConv2D, self).__init__()
        
        # block 1
        self.conv1 = nn.Conv2d(
            in_channels=inChannels, 
            out_channels=outChannels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.norm1 = nn.BatchNorm2d(outChannels)
        self.relu1 = nn.ReLU()
        
        # block 2
        self.conv2 = nn.Conv2d(
            in_channels=outChannels, 
            out_channels=outChannels, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False)
        self.norm2 = nn.BatchNorm2d(outChannels)
        self.relu2 = nn.ReLU()

    # function:    forward
    # description: passes an input through the network and returns the result
    # inputs:      x - the input to be passed through the network
    # outputs:     the result of x being passed through the network
    def forward(self, x):
        x = self.conv1(x)
        
        # BatchNorm2D expects a 4D input but x is only 3D, so we need to add an extra dimension
        x = torch.unsqueeze(x, dim=0)
        x = self.norm1(x)
        
        # remove the extra dimension before proceeding
        x = torch.squeeze(x, dim=0)
        x = self.relu1(x)
        
        x = self.conv2(x)
        
        # BatchNorm2D expects a 4D input but x is only 3D, so we need to add an extra dimension
        x = torch.unsqueeze(x, dim=0)
        x = self.norm2(x)
        
        # remove the extra dimension before proceeding
        x = torch.squeeze(x, dim=0)
        x = self.relu2(x)
        
        # return the result
        return x
    
# -------------------------------------------------------------------------------------- #
# Class:       Generator                                                                 #
# Description: This class defines the generator of our cycle-GAN. We chose to implement  #
#              our generator as a U-Net CNN, which uses an encoder and a decoder, each   #
#              containing five residual blocks.                                          #
# -------------------------------------------------------------------------------------- #
class Generator(nn.Module):
    # function:    init
    # description: initializes class attributes (defines layers)
    # inputs:      none
    # outputs:     none
    def __init__(self):
        super(Generator, self).__init__()

        # ----- encoder layers ----- #
        # encoder block 1
        self.dc1 = DoubleConv2D(inChannels=1, outChannels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # encoder block 2
        self.dc2 = DoubleConv2D(inChannels=64, outChannels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # encoder block 3
        self.dc3 = DoubleConv2D(inChannels=128, outChannels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # encoder block 4
        self.dc4 = DoubleConv2D(inChannels=256, outChannels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # encoder block 5
        self.dc5 = DoubleConv2D(inChannels=512, outChannels=1024)
            
        # ----- decoder layers ----- #
        # decoder block 1
        self.conv5T = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)   
        
        # decoder block 2
        self.dc6 = DoubleConv2D(inChannels=1024, outChannels=512)
        self.conv6T = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2) 
        
        # decoder block 3
        self.dc7 = DoubleConv2D(inChannels=512, outChannels=256)
        self.conv7T = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)  
        
        # decoder block 4
        self.dc8 = DoubleConv2D(inChannels=256, outChannels=128)
        self.conv8T = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)  
        
        # decoder block 5
        self.dc9 = DoubleConv2D(inChannels=128, outChannels=64)
        self.conv9 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        
    # function:    forward
    # description: passes an input forward through the network
    # inputs:      x - the input to be passed through the network
    # outputs:     returns the result of passing x through the network
    def forward(self, x):
        # ----- contraction path ----- #
        # encoder block 1
        dc1out = self.dc1(x)
        pool1out = self.pool1(dc1out)

        # encoder block 2
        dc2out = self.dc2(pool1out)
        pool2out = self.pool2(dc2out)

        # encoder block 3
        dc3out = self.dc3(pool2out)
        pool3out = self.pool3(dc3out)
        
        # encoder block 4
        dc4out = self.dc4(pool3out)
        pool4out = self.pool4(dc4out)
        
        # encoder block 5
        dc5out = self.dc5(pool4out)
        
        # ----- expansion path ----- #
        # decoder block 1
        conv5Tout = self.conv5T(dc5out)
        
        # skip connection
        cat5out = torch.cat((dc4out, conv5Tout), dim=0)
        
        # decoder block 2
        dc6out = self.dc6(cat5out)
        conv6Tout = self.conv6T(dc6out)
        
        # skip connection
        cat6out = torch.cat((dc3out, conv6Tout), dim=0)
        
        # decoder block 3
        dc7out = self.dc7(cat6out)
        conv7Tout = self.conv7T(dc7out)
        
        # skip connection
        cat7out = torch.cat((dc2out, conv7Tout), dim=0)
        
        # decoder block 4
        dc8out = self.dc8(cat7out)
        conv8Tout = self.conv8T(dc8out)
        
        # skip connection
        cat8out = torch.cat((dc1out, conv8Tout), dim=0)
        
        # decoder block 5
        dc9out = self.dc9(cat8out)
        conv9out = self.conv9(dc9out)

        # return result
        return conv9out
        