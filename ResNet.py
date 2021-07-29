import torch.nn as nn
import math


class ResNet(nn.Module):

    def __init__(self, img_size, channel_in=64, channel_out=512, bb_chain=(3, 4, 6, 3), bb_depth=2, out_classes=10):
        """
        The Residual Network is based on:
        He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        :param img_size: size of the image pixel vise
        :param channel_in: amount of channel at the begin of the residual blocks
        :param channel_out: amount of channel at the end of the residual blocks
        :param bb_chain: Number of risidual blocks for each channel depth block
        :param bb_depth: Number of convolution for each building block
        :param out_classes: amount of different objects to predict
        """
        super().__init__()
        self.in_size = (1, channel_in)
        self.bb_depth = bb_depth
        self.channels = [(channel_in, channel_in)]
        # get the number of channels in and out for each channel block given the amount
        # of channel at initialization and at the end
        self.channels.extend([
            (channel_in*2**i, channel_in*2**i*2) for i in
            range(int(math.log(channel_out/channel_in) / math.log(2)))])
        # in features for the linear layer
        # max channels times the dimensions of the picture after conv und max. pool
        # a kernel size of 7 with a padding of 1,
        # reducing the original image by 4 (kernel_size-1-2*padding).
        # Each max pool of kernel 2 halves the remaining image size
        self.linear_in = channel_out * \
                         int((img_size[0]-4) / 2**2) * \
                         int((img_size[1]-4) / 2**2)
        self.bb_chain = bb_chain

        # layer building
        self.layer = [nn.Conv2d(*self.in_size,
                                kernel_size=7,
                                padding=1),
                      nn.MaxPool2d(2, 2),
                      nn.ReLU()]

        # adding of each building block while bb_chains provides the information
        # how many building blocks are given for a given channel size
        self.layer.extend([BuildingBlock(channels=c, bb_chain=i, bb_depth=self.bb_depth)
                           for _, c in enumerate(self.channels)
                           for i in range(bb_chain[_])])

        self.layer.extend([nn.MaxPool2d(2, 2),
                           nn.Flatten(),
                           nn.Linear(in_features=self.linear_in,
                                     out_features=out_classes),
                           nn.Softmax(dim=-1)])

        self.layer = nn.Sequential(*self.layer)

    def forward(self, x):
        return self.layer(x)


class BuildingBlock(nn.Module):

    def __init__(self, channels, bb_chain, bb_depth):
        """
        A standard building blocks of the residual learning consist of:
        conv + batch norm + relu + conv + batch norm (+identity) + relu
        The skip connection is added on top of the last layer after batch norm
        In chase of differences of the channels in/out a linear projection is
        used otherwise the identity mapping is used

        :param channels: The in and out dimensions
        :param bb_chain: Number of risidual blocks for each channel depth block
        :param bb_depth: Number of convolution for each building block
        """
        super().__init__()
        self.bb_depth = bb_depth

        self.first_layer = nn.Conv2d(in_channels=channels[bb_chain > 0],
                                     out_channels=channels[1],
                                     kernel_size=3,
                                     padding=1)
        self.follow_up = nn.Conv2d(channels[1], channels[1], kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU(inplace=True)

        self.identity = nn.Identity() if bb_chain != 0 else \
            nn.Conv2d(*channels, 1)

    def forward(self, x):

        skip_in = x

        out = self.first_layer(x)
        out = self.batch_norm(out)
        out = self.relu(out)

        if self.bb_depth > 2:
            for i in range(self.bb_depth-2):
                out = self.follow_up(out)
                out = self.batch_norm(out)
                out = self.relu(out)

        out = self.follow_up(out)
        out = self.batch_norm(out)

        out += self.identity(skip_in)
        out = self.relu(out)

        return out





