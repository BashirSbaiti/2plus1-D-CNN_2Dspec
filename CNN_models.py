import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict


def updateShape(inShape: list, convShape: tuple, outChannels: int, paddings=0, strides=1):
    if type(strides) is not tuple and type(strides) is not list:
        strides = [strides] * len(inShape)
    if type(paddings) is not tuple and type(paddings) is not list:
        paddings = [paddings] * len(inShape)
    if paddings == "same" and 1 in strides:
        raise Exception("Padding cannot be same if strides are not one.")

    outShape = []
    inShapeNoCh = inShape[1:].copy()  # pop off channel information
    for inDim, convDim, stride, pad in zip(inShapeNoCh, convShape, strides, paddings):
        if pad == "same":
            outDim = inDim
        else:
            nSteps = len(range(convDim - 1 - pad, inDim + pad, stride))
            outDim = nSteps
        outShape.append(outDim)
    outShape.insert(0, outChannels)
    return outShape


class Stem2plus1D(nn.Sequential):
    def __init__(self, inChannels, outChannels, convShape) -> None:
        spacePad = int(convShape[1] / 2)
        timePad = int(convShape[0] / 2)
        super().__init__(
            nn.Conv3d(inChannels, outChannels, kernel_size=(1, convShape[1], convShape[2]), stride=(1, 2, 2),
                      padding=(0, spacePad, spacePad)),
            nn.BatchNorm3d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv3d(outChannels, outChannels, kernel_size=(convShape[0], 1, 1), stride=(2, 1, 1),
                      padding=(timePad, 0, 0)),
            nn.BatchNorm3d(outChannels),
            nn.ReLU(inplace=True),
        )


class Stem3D(nn.Sequential):
    def __init__(self, inChannels, outChannels, convShape) -> None:
        spacePad = int(convShape[1] / 2)
        timePad = int(convShape[0] / 2)
        super().__init__(
            nn.Conv3d(inChannels, outChannels, kernel_size=(convShape[0], convShape[1], convShape[2]), stride=(2, 2, 2),
                      padding=(timePad, spacePad, spacePad)),
            nn.BatchNorm3d(outChannels),
            nn.ReLU(inplace=True),
        )


class ConvBlock2plus1D(nn.Module):
    def __init__(self, convShape2d: tuple, tempConv: int, channelsIn: int, channelsOut: int, stride: int,
                 padding: tuple):
        """
        :param convShape: shape of convolutions to do in (w1, w3)
        :param tempConv: temporal conv length to do (depthwise in t2)
        :param channelsIn: number of input channels
        :param channelsOut: number of output channels
        """
        super().__init__()

        t = tempConv
        d = convShape2d[0]
        N = channelsIn
        N1 = channelsOut
        intermed = (t * (d ** 2) * N * N1) / (
                (d ** 2) * N + t * N1)  # used in https://arxiv.org/pdf/1711.11248.pdf (pg 4)

        self.spacialConv = nn.Conv3d(in_channels=channelsIn, out_channels=channelsOut,
                                     kernel_size=(1, convShape2d[0], convShape2d[1]), stride=(1, stride, stride),
                                     padding=(0, padding[1], padding[2]))
        self.bn1 = nn.BatchNorm3d(channelsOut)
        self.relu1 = nn.ReLU()
        self.temporalConv = nn.Conv3d(in_channels=channelsOut, out_channels=channelsOut,
                                      kernel_size=(tempConv, 1, 1), stride=(stride, 1, 1),
                                      padding=(padding[0], 0, 0))
        self.bn2 = nn.BatchNorm3d(channelsOut)
        self.relu2 = nn.ReLU()
        # self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.spacialConv(x)
        x = self.bn1(x)
        x = self.relu1(x)  # can go with or without, but paper says this is good due to extra nonlinearity
        x = self.temporalConv(x)
        x = self.bn2(x)
        out = self.relu2(x)
        # out = self.mp(x)
        return out


class ConvBlock3D(nn.Module):
    def __init__(self, convShape: tuple, channelsIn: int, channelsOut: int, stride: int,
                 padding: tuple):
        """
        :param convShape: shape of convolutions to do in (t2, w1, w3)
        :param channelsIn: number of input channels
        :param channelsOut: number of output channels
        """
        super().__init__()

        self.conv = nn.Conv3d(in_channels=channelsIn, out_channels=channelsOut,
                              kernel_size=convShape, stride=(stride, stride, stride),
                              padding=(padding[0], padding[1], padding[2]))
        self.bn = nn.BatchNorm3d(channelsOut)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out


class CNN2plus1Le(nn.Module):
    def __init__(self, inShape: tuple, convShape: tuple, stemConvShape: tuple, numClasses: int, convBlocks: int,
                 outChannels: tuple,
                 strides: tuple, t2pads: tuple,
                 useGAP=False):
        """
        :param inShape: shape of input data cube (nChannels, t2, w1, w3)
        :param convShape: shape of convolutions to do in (t2, w1, w3)
        :param stemConvShape: shape of convolutions in the first layer (t2, w1, w3)
        :param numClasses: number of final output classes
        :param convBlocks: number of (conv -> relu -> ...) blocks
        :param outChannels: number of output channels for each convolution, input data is assumed to be 1 channel
        :param useGAP: use global average pooling instead of flattening when connecting to fc layer.
        :param t2pads: number to zero pad on each size in t2 dimension
        """
        super().__init__()
        self.useGap = useGAP
        t2c, w1c, w3c = convShape
        currentShape = list(inShape)  # keep track of data shape throughout process
        self.outChannels = outChannels
        self.t2Pads = t2pads
        self.layers = OrderedDict()

        if len(self.outChannels) != convBlocks:
            raise Exception("Number of out channels must equal number of conv blocks requested, including the stem")
        if len(strides) != convBlocks - 1:
            raise Exception("Number of strides provided must equal number of (non-stem) conv blocks requested")
        if len(t2pads) != convBlocks - 1:
            raise Exception("Number of padding values provided must equal number of (non-stem) conv blocks requested")

        t2Collapsed = False

        # initialize a variable number of convBlocks (conv -> relu -> ...), while keeping track of shape

        # stem first
        currentShape = updateShape(currentShape, stemConvShape, self.outChannels[0], strides=(2, 2, 2),
                                   paddings=[i // 2 for i in stemConvShape])
        self.layers[f"2plus1D Stem"] = Stem2plus1D(inShape[0], self.outChannels[0], stemConvShape)

        # then rest of network
        for i in range(2, convBlocks + 1):
            nextShape = updateShape(currentShape, (t2c, w1c, w3c), self.outChannels[i - 1], strides=strides[i - 2],
                                    paddings=(self.t2Pads[i - 2], 0, 0))
            # check if spacial conv will overflow input size or if t2 dimension has already been collapsed to one
            # (temporal conv makes no sense in this case!)
            if np.prod(nextShape[2:]) > 0 and not t2Collapsed:
                if nextShape[1] <= 0:
                    t2c = currentShape[1]
                    nextShape = updateShape(currentShape, (t2c, w1c, w3c), self.outChannels[i - 1],
                                            strides=strides[i - 2], paddings=(self.t2Pads[i - 2], 0, 0))
                    print(
                        f"Warning: t2 conv overflow was avoided by changing temporal conv from {convShape[0]} to {t2c}")
                    t2Collapsed = True
                currentShape = nextShape
                inCh = self.outChannels[i - 2] if i - 2 >= 0 else inShape[0]

                spacialConv = (w1c, w3c)
                temporalConv = t2c
                self.layers[f"Conv Block {i}"] = ConvBlock2plus1D(spacialConv, temporalConv, inCh,
                                                                  self.outChannels[i - 1],
                                                                  strides[i - 2], padding=(self.t2Pads[i - 2], 0, 0))
            else:
                print(
                    f"Warning: convolution {i} was skipped because conv size {convShape} > input shape {nextShape[1:]}.")

        if useGAP:
            self.layers["GAP"] = nn.AdaptiveAvgPool3d(1)
            self.layers["Reshape"] = nn.Flatten()
            self.layers["FC"] = nn.Linear(in_features=self.outChannels[-1], out_features=numClasses)
        else:
            self.layers["flatten"] = nn.Flatten()  # default params will flatten all but batch giving shape (batch, x)
            self.layers["FC1"] = nn.Linear(in_features=int(np.prod(currentShape)), out_features=1024)
            self.layers[f"ReLU_FC"] = nn.ReLU()
            #self.layers["Dropout1"] = nn.Dropout(0.2)
            self.layers[f"FC2"] = nn.Linear(in_features=1024, out_features=numClasses)

        self.fullNet = nn.Sequential(self.layers)

    def forward(self, x):
        output = self.fullNet(x)
        return output


class CNN3dLe(nn.Module):
    def __init__(self, inShape: tuple, convShape: tuple, stemConvShape: tuple, numClasses: int, convBlocks: int,
                 outChannels: tuple,
                 strides: tuple, t2pads: tuple,
                 useGAP=False):
        """
        :param inShape: shape of input data cube (nChannels, t2, w1, w3)
        :param convShape: shape of convolutions to do in (t2, w1, w3)
        :param stemConvShape: shape of convolutions in the first layer (t2, w1, w3)
        :param numClasses: number of final output classes
        :param convBlocks: number of (conv -> relu -> ...) blocks
        :param outChannels: number of output channels for each convolution, input data is assumed to be 1 channel
        :param useGAP: use global average pooling instead of flattening when connecting to fc layer.
        :param t2pads: number to zero pad on each size in t2 dimension
        """
        super().__init__()
        self.useGap = useGAP
        t2c, w1c, w3c = convShape
        currentShape = list(inShape)  # keep track of data shape throughout process
        self.outChannels = outChannels
        self.t2Pads = t2pads
        self.layers = OrderedDict()

        if len(self.outChannels) != convBlocks:
            raise Exception("Number of out channels must equal number of conv blocks requested, including the stem")
        if len(strides) != convBlocks - 1:
            raise Exception("Number of strides provided must equal number of (non-stem) conv blocks requested")
        if len(t2pads) != convBlocks - 1:
            raise Exception("Number of padding values provided must equal number of (non-stem) conv blocks requested")

        t2Collapsed = False

        # initialize a variable number of convBlocks (conv -> relu -> ...), while keeping track of shape

        # stem first
        currentShape = updateShape(currentShape, stemConvShape, self.outChannels[0], strides=(2, 2, 2),
                                   paddings=[i // 2 for i in stemConvShape])
        self.layers[f"3D Stem"] = Stem3D(inShape[0], self.outChannels[0], stemConvShape)

        # then rest of network
        for i in range(2, convBlocks + 1):
            nextShape = updateShape(currentShape, (t2c, w1c, w3c), self.outChannels[i - 1], strides=strides[i - 2],
                                    paddings=(self.t2Pads[i - 2], 0, 0))
            # check if conv will overflow input size or if t2 dimension has already been collapsed to one
            # (temporal conv makes no sense in this case!)
            if np.prod(nextShape[2:]) > 0 and not t2Collapsed:
                if nextShape[1] <= 0:
                    t2c = currentShape[1]
                    nextShape = updateShape(currentShape, (t2c, w1c, w3c), self.outChannels[i - 1],
                                            strides=strides[i - 2], paddings=(self.t2Pads[i - 2], 0, 0))
                    print(
                        f"Warning: t2 conv overflow was avoided by changing conv in t2 from {convShape[0]} to {t2c}")
                    t2Collapsed = True
                currentShape = nextShape
                inCh = self.outChannels[i - 2] if i - 2 >= 0 else inShape[0]

                self.layers[f"Conv Block {i}"] = ConvBlock3D((t2c, w1c, w3c), inCh,
                                                             self.outChannels[i - 1],
                                                             strides[i - 2], padding=(self.t2Pads[i - 2], 0, 0))
            else:
                print(
                    f"Warning: convolution {i} was skipped because conv size {convShape} > input shape {nextShape[1:]}.")

        if useGAP:
            self.layers["GAP"] = nn.AdaptiveAvgPool3d(1)
            self.layers["Reshape"] = nn.Flatten()
            self.layers["FC"] = nn.Linear(in_features=self.outChannels[-1], out_features=numClasses)
        else:
            self.layers["flatten"] = nn.Flatten()  # default params will flatten all but batch giving shape (batch, x)
            self.layers["FC1"] = nn.Linear(in_features=int(np.prod(currentShape)), out_features=1024)
            self.layers[f"ReLU_FC"] = nn.ReLU()
            #self.layers["Dropout1"] = nn.Dropout(0.2)
            self.layers[f"FC2"] = nn.Linear(in_features=1024, out_features=numClasses)

        self.fullNet = nn.Sequential(self.layers)

    def forward(self, x):
        output = self.fullNet(x)
        return output
