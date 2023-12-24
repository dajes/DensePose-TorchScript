# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict, List

import fvcore.nn.weight_init as weight_init
import numpy as np
import torch
import torch.nn as nn

from detectron2.layers import Conv2d, get_norm
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from .. import (
    build_densepose_data_filter,
    build_densepose_embedder,
    build_densepose_head,
    build_densepose_predictor,
    densepose_inference,
)


class Decoder(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, dict], in_features):
        super(Decoder, self).__init__()

        # fmt: off
        self.in_features = in_features
        feature_strides = {k: v['stride'] for k, v in input_shape.items()}
        feature_channels = {k: v['channels'] for k, v in input_shape.items()}
        num_classes = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NUM_CLASSES
        conv_dims = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_CONV_DIMS
        self.common_stride = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_COMMON_STRIDE
        norm = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_NORM
        # fmt: on

        scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, conv_dims),
                    activation=nn.ReLU(),
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, scale_heads[-1])
        self.scale_heads = nn.ModuleList(scale_heads)
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features: List[torch.Tensor]):
        x = features[0]
        for i, head in enumerate(self.scale_heads):
            if i == 0:
                x = head(features[i])
            else:
                x = x + head(features[i])
        x = self.predictor(x)
        return x


@ROI_HEADS_REGISTRY.register()
class DensePoseROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains an addition of DensePose head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_densepose_head(cfg, input_shape)

    def _init_densepose_head(self, cfg, input_shape):
        # fmt: off
        self.densepose_on = cfg.MODEL.DENSEPOSE_ON
        if not self.densepose_on:
            return
        self.densepose_data_filter = build_densepose_data_filter(cfg)
        dp_pooler_resolution = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION
        dp_pooler_sampling_ratio = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO
        dp_pooler_type = cfg.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE
        self.use_decoder = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECODER_ON
        # fmt: on
        if self.use_decoder:
            dp_pooler_scales = (1.0 / input_shape[self.in_features[0]]['stride'],)
        else:
            dp_pooler_scales = tuple(1.0 / input_shape[k]['stride'] for k in self.in_features)
        in_channels = [input_shape[f]['channels'] for f in self.in_features][0]

        if self.use_decoder:
            self.decoder = Decoder(cfg, input_shape, self.in_features)
        else:
            self.decoder = None

        self.densepose_pooler = ROIPooler(
            output_size=dp_pooler_resolution,
            scales=dp_pooler_scales,
            sampling_ratio=dp_pooler_sampling_ratio,
            pooler_type=dp_pooler_type,
        )
        self.densepose_head = build_densepose_head(cfg, in_channels)
        self.densepose_predictor = build_densepose_predictor(
            cfg, self.densepose_head.n_out_channels
        )
        self.embedder = build_densepose_embedder(cfg)

    def _forward_densepose(self, features: Dict[str, torch.Tensor], instances: List[Dict[str, torch.Tensor]]):
        """
        Forward logic of the densepose prediction branch.

        Args:
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            instances (list): length `N` list of. The i-th
                contains instances for the i-th input image,
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "densepose" and return it.
        """
        if not self.densepose_on:
            return instances

        features_list = [features[f] for f in self.in_features]
        pred_boxes = [x['pred_boxes'] for x in instances]

        if self.decoder is not None:
            features_list = [self.decoder(features_list)]

        features_dp = self.densepose_pooler(features_list, pred_boxes)
        densepose_head_outputs = self.densepose_head(features_dp)
        densepose_predictor_outputs = self.densepose_predictor(densepose_head_outputs)

        densepose_inference(densepose_predictor_outputs, instances)
        return instances

    def forward_with_given_boxes(
            self, features: Dict[str, torch.Tensor], instances: List[Dict[str, torch.Tensor]]
    ):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list):
                the same objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        instances = self._forward_densepose(features, instances)
        return instances
