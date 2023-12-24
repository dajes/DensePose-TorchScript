# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

"""
This file contains components with some default boilerplate logic user may need
in training / testing. They will not work for everyone, but many users may find them useful.

The behavior of functions/classes in this file is subject to change,
since they are meant to represent the "common default behavior" people need in their projects.
"""

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

__all__ = ["DefaultPredictor", ]


class DefaultPredictor(nn.Module):
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.min_size = cfg.INPUT.MIN_SIZE_TEST
        self.max_size = cfg.INPUT.MAX_SIZE_TEST

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def forward(self, original_image: torch.Tensor, bgr: bool = True):
        """
        Args:
            original_image (torch.Tensor): an image of shape (H, W, C)
            bgr (bool): whether the original image is in BGR color format.
                Otherwise it should be RGB format.

        Returns:
            predictions (dict):
                the output of the model for one image only.
        """
        if original_image.shape[2] == 3:
            original_image = original_image.permute(2, 0, 1)
        else:
            assert original_image.shape[0] == 3, (
                'Only 3 channels expected either in HWC or CHW format, got {}'.format(original_image.shape))

        if self.input_format == "RGB" and bgr:
            original_image = original_image.flip(0)

        height = original_image.shape[1]
        width = original_image.shape[2]
        k = min(self.min_size / min(height, width), self.max_size / max(height, width))

        image = F.interpolate(original_image[None], scale_factor=k, mode="bilinear", align_corners=False)[0]
        inputs = {
            "image": image,
            "height": torch.tensor(height, dtype=torch.int64),
            "width": torch.tensor(width, dtype=torch.int64)
        }
        with torch.no_grad():
            predictions = self.model([inputs])[0]
        return predictions
