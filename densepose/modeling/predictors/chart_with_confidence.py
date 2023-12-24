# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Any

import torch

from detectron2.config import CfgNode
from detectron2.layers import ConvTranspose2d
from . import DensePoseChartPredictor
from .registry import DENSEPOSE_PREDICTOR_REGISTRY
from ..confidence import DensePoseConfidenceModelConfig, DensePoseUVConfidenceType


@DENSEPOSE_PREDICTOR_REGISTRY.register()
class DensePoseChartWithConfidencePredictor(DensePoseChartPredictor):
    """
    Predictor that combines chart and chart confidence estimation
    """

    """
    Predictor contains the last layers of a DensePose model that take DensePose head
    outputs as an input and produce model outputs. Confidence predictor mixin is used
    to generate confidences for segmentation and UV tensors estimated by some
    base predictor. Several assumptions need to hold for the base predictor:
    1) the `forward` method must return SIUV tuple as the first result (
        S = coarse segmentation, I = fine segmentation, U and V are intrinsic
        chart coordinates)
    2) `interp2d` method must be defined to perform bilinear interpolation;
        the same method is typically used for SIUV and confidences
    Confidence predictor mixin provides confidence estimates, as described in:
        N. Neverova et al., Correlated Uncertainty for Learning Dense Correspondences
            from Noisy Labels, NeurIPS 2019
        A. Sanakoyeu et al., Transferring Dense Pose to Proximal Animal Classes, CVPR 2020
    """

    def __init__(self, cfg: CfgNode, input_channels: int):
        """
        Initialize confidence predictor using configuration options.

        Args:
            cfg (CfgNode): configuration options
            input_channels (int): number of input channels
        """
        # we rely on base predictor to call nn.Module.__init__
        super().__init__(cfg, input_channels)  # pyre-ignore[19]
        self.confidence_model_cfg = DensePoseConfidenceModelConfig.from_cfg(cfg)
        self._initialize_confidence_estimation_layers(cfg, input_channels)
        self._registry = {}

    def _initialize_confidence_estimation_layers(self, cfg: CfgNode, dim_in: int):
        """
        Initialize confidence estimation layers based on configuration options

        Args:
            cfg (CfgNode): configuration options
            dim_in (int): number of input channels
        """
        dim_out_patches = cfg.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES + 1
        kernel_size = cfg.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL
        if self.confidence_model_cfg.uv_confidence.enabled:
            if self.confidence_model_cfg.uv_confidence.type == DensePoseUVConfidenceType.IID_ISO:
                self.sigma_2_lowres = ConvTranspose2d(  # pyre-ignore[16]
                    dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
                )
            elif (
                    self.confidence_model_cfg.uv_confidence.type
                    == DensePoseUVConfidenceType.INDEP_ANISO
            ):
                self.sigma_2_lowres = ConvTranspose2d(
                    dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
                )
                self.kappa_u_lowres = ConvTranspose2d(  # pyre-ignore[16]
                    dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
                )
                self.kappa_v_lowres = ConvTranspose2d(  # pyre-ignore[16]
                    dim_in, dim_out_patches, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
                )
            else:
                raise ValueError(
                    f"Unknown confidence model type: "
                    f"{self.confidence_model_cfg.confidence_model_type}"
                )
        if self.confidence_model_cfg.segm_confidence.enabled:
            self.fine_segm_confidence_lowres = ConvTranspose2d(  # pyre-ignore[16]
                dim_in, 1, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
            )
            self.coarse_segm_confidence_lowres = ConvTranspose2d(  # pyre-ignore[16]
                dim_in, 1, kernel_size, stride=2, padding=int(kernel_size / 2 - 1)
            )

    def forward(self, head_outputs: torch.Tensor):
        """
        Perform forward operation on head outputs used as inputs for the predictor.
        Calls forward method from the base predictor and uses its outputs to compute
        confidences.

        Args:
            head_outputs (Tensor): head outputs used as predictor inputs
        Return:
            An instance of outputs with confidences,
            see `decorate_predictor_output_class_with_confidences`
        """
        output = dict(
            coarse_segm=self.interp2d(self.ann_index_lowres(head_outputs)),
            fine_segm=self.interp2d(self.index_uv_lowres(head_outputs)),
            u=self.interp2d(self.u_lowres(head_outputs)),
            v=self.interp2d(self.v_lowres(head_outputs)),
        )
        return output

    def _create_output_instance(self, base_predictor_outputs: Any):
        """
        Create an instance of predictor outputs by copying the outputs from the
        base predictor and initializing confidence

        Args:
            base_predictor_outputs: an instance of base predictor outputs
                (the outputs type is assumed to be a dataclass)
        Return:
           An instance of outputs with confidences
        """
        return base_predictor_outputs
