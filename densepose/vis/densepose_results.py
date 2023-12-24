from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from detectron2.structures import Boxes


class DensePoseResultsVisualizer:
    def visualize(
            self,
            image_bgr: np.ndarray,
            results_and_boxes_xywh: Tuple[Optional[list], Optional[Boxes]],
    ) -> np.ndarray:
        densepose_result, boxes_xywh = results_and_boxes_xywh
        if densepose_result is None or boxes_xywh is None:
            return image_bgr

        boxes_xywh = boxes_xywh.cpu().numpy()
        context = self.create_visualization_context(image_bgr)
        for i, result in enumerate(densepose_result):
            iuv_array = torch.cat(
                (result.labels[None].type(torch.float32), result.uv * 255.0)
            ).type(torch.uint8)
            self.visualize_iuv_arr(context, iuv_array.cpu().numpy(), boxes_xywh[i])
        image_bgr = self.context_to_image_bgr(context)
        return image_bgr

    def create_visualization_context(self, image_bgr):
        return image_bgr

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh) -> None:
        pass

    def context_to_image_bgr(self, context):
        return context

    def get_image_bgr_from_context(self, context):
        return context


class MatrixVisualizer:
    """
    Base visualizer for matrix data
    """

    def __init__(
            self,
            inplace=True,
            cmap=cv2.COLORMAP_PARULA,
            val_scale=1.0,
            alpha=0.7,
            interp_method_matrix=cv2.INTER_LINEAR,
            interp_method_mask=cv2.INTER_NEAREST,
    ):
        self.inplace = inplace
        self.cmap = cmap
        self.val_scale = val_scale
        self.alpha = alpha
        self.interp_method_matrix = interp_method_matrix
        self.interp_method_mask = interp_method_mask

    def visualize(self, image_bgr, mask, matrix, bbox_xywh):
        self._check_image(image_bgr)
        self._check_mask_matrix(mask, matrix)
        if self.inplace:
            image_target_bgr = image_bgr
        else:
            image_target_bgr = image_bgr * 0
        x, y, w, h = [int(v) for v in bbox_xywh]
        if w <= 0 or h <= 0:
            return image_bgr
        mask, matrix = self._resize(mask, matrix, w, h)
        mask_bg = np.tile((mask == 0)[:, :, np.newaxis], [1, 1, 3])
        matrix_scaled = matrix.astype(np.float32) * self.val_scale
        matrix_scaled_8u = matrix_scaled.clip(0, 255).astype(np.uint8)
        matrix_vis = cv2.applyColorMap(matrix_scaled_8u, self.cmap)
        matrix_vis[mask_bg] = image_target_bgr[y: y + h, x: x + w, :][mask_bg]
        image_target_bgr[y: y + h, x: x + w, :] = (
                image_target_bgr[y: y + h, x: x + w, :] * (1.0 - self.alpha) + matrix_vis * self.alpha
        )
        return image_target_bgr.astype(np.uint8)

    def _resize(self, mask, matrix, w, h):
        if (w != mask.shape[1]) or (h != mask.shape[0]):
            mask = cv2.resize(mask, (w, h), self.interp_method_mask)
        if (w != matrix.shape[1]) or (h != matrix.shape[0]):
            matrix = cv2.resize(matrix, (w, h), self.interp_method_matrix)
        return mask, matrix

    def _check_image(self, image_rgb):
        assert len(image_rgb.shape) == 3
        assert image_rgb.shape[2] == 3
        assert image_rgb.dtype == np.uint8

    def _check_mask_matrix(self, mask, matrix):
        assert len(matrix.shape) == 2
        assert len(mask.shape) == 2
        assert mask.dtype == np.uint8


class DensePoseMaskedColormapResultsVisualizer(DensePoseResultsVisualizer):
    def __init__(
            self,
            data_extractor,
            segm_extractor,
            inplace=True,
            cmap=cv2.COLORMAP_PARULA,
            alpha=0.7,
            val_scale=1.0,
            **kwargs,
    ):
        self.mask_visualizer = MatrixVisualizer(
            inplace=inplace, cmap=cmap, val_scale=val_scale, alpha=alpha
        )
        self.data_extractor = data_extractor
        self.segm_extractor = segm_extractor

    def context_to_image_bgr(self, context):
        return context

    def visualize_iuv_arr(self, context, iuv_arr: np.ndarray, bbox_xywh) -> None:
        image_bgr = self.get_image_bgr_from_context(context)
        matrix = self.data_extractor(iuv_arr)
        segm = self.segm_extractor(iuv_arr)
        mask = np.zeros(matrix.shape, dtype=np.uint8)
        mask[segm > 0] = 1
        image_bgr = self.mask_visualizer.visualize(image_bgr, mask, matrix, bbox_xywh)


def _extract_i_from_iuvarr(iuv_arr):
    return iuv_arr[0, :, :]


class DensePoseResultsFineSegmentationVisualizer(DensePoseMaskedColormapResultsVisualizer):
    N_PART_LABELS = 24

    def __init__(self, inplace=True, cmap=cv2.COLORMAP_PARULA, alpha=0.7, **kwargs):
        super(DensePoseResultsFineSegmentationVisualizer, self).__init__(
            _extract_i_from_iuvarr,
            _extract_i_from_iuvarr,
            inplace,
            cmap,
            alpha,
            val_scale=255.0 / self.N_PART_LABELS,
            **kwargs,
        )
