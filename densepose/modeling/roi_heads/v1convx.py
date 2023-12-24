# Copyright (c) Facebook, Inc. and its affiliates.

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import CfgNode
from detectron2.layers import Conv2d
from .registry import ROI_DENSEPOSE_HEAD_REGISTRY


@ROI_DENSEPOSE_HEAD_REGISTRY.register()
class DensePoseV1ConvXHead(nn.Module):
    """
    Fully convolutional DensePose head.
    """

    def __init__(self, cfg: CfgNode, input_channels: int):
        """
        Initialize DensePose fully convolutional head

        Args:
            cfg (CfgNode): configuration options
            input_channels (int): number of input channels
        """
        super(DensePoseV1ConvXHead, self).__init__()
        # fmt: off
        hidden_dim = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL
        self.n_stacked_convs = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS
        # fmt: on
        pad_size = kernel_size // 2
        n_channels = input_channels
        stacked_convs = []
        for i in range(self.n_stacked_convs):
            layer = Conv2d(n_channels, hidden_dim, kernel_size, stride=1, padding=pad_size)
            layer_name = self._get_layer_name(i)
            self.add_module(layer_name, layer)
            n_channels = hidden_dim
            stacked_convs.append(layer)
        self.stacked_convs = nn.ModuleList(stacked_convs)
        self.n_out_channels = n_channels

    def forward(self, features: torch.Tensor):
        """
        Apply DensePose fully convolutional head to the input features

        Args:
            features (tensor): input features
        Result:
            A tensor of DensePose head outputs
        """
        x = features
        output = x
        for i, layer in enumerate(self.stacked_convs):
            x = layer(x)
            x = F.relu(x)
            output = x
        return output

    def _get_layer_name(self, i: int):
        layer_name = "body_conv_fcn{}".format(i + 1)
        return layer_name
