from typing import Any, Tuple, Dict
from typing import Optional

import cv2
import numpy as np
import torch
from torch.nn import functional as F


def resample_fine(fine_segm: torch.Tensor, coarse_segm: torch.Tensor, box_xywh_abs):
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    coarse_segm_bbox = F.interpolate(coarse_segm, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
    m = (coarse_segm_bbox > 0).long()
    labels = F.interpolate(fine_segm, (h, w), mode="bilinear", align_corners=False).argmax(dim=1) * m
    return labels


def resample_uv_tensors_to_bbox(u: torch.Tensor, v: torch.Tensor, labels: torch.Tensor, box_xywh_abs) -> torch.Tensor:
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    u_bbox = F.interpolate(u, (h, w), mode="bilinear", align_corners=False).float()
    v_bbox = F.interpolate(v, (h, w), mode="bilinear", align_corners=False).float()
    uv = torch.zeros([2, h, w], dtype=torch.float32, device=u.device)
    for part_id in range(1, u_bbox.size(1)):
        uv[0][labels == part_id] = u_bbox[0, part_id][labels == part_id]
        uv[1][labels == part_id] = v_bbox[0, part_id][labels == part_id]
    return uv


def predictor_output_to_result(predictor_output: Dict[str, torch.Tensor], box_xywh):
    box_xywh = box_xywh.long().tolist()
    labels = resample_fine(predictor_output['fine_segm'], predictor_output['coarse_segm'], box_xywh).squeeze(0)
    uv = resample_uv_tensors_to_bbox(predictor_output['u'], predictor_output['v'], labels, box_xywh)
    return dict(labels=labels, uv=uv)


def extract_boxes_xywh_from_instances(instances):
    boxes_xywh = instances['pred_boxes'].clone()
    boxes_xywh[:, 2:] -= boxes_xywh[:, :2]
    return boxes_xywh


class DensePoseResultExtractor:
    def __call__(self, instances) -> Tuple[Any, Optional[torch.Tensor]]:
        boxes_xyxy = instances['pred_boxes']
        boxes_xywh = extract_boxes_xywh_from_instances(instances)
        results = [predictor_output_to_result({
            'coarse_segm': instances['pred_densepose_coarse_segm'][i].unsqueeze(0),
            'fine_segm': instances['pred_densepose_fine_segm'][i].unsqueeze(0),
            'u': instances['pred_densepose_u'][i].unsqueeze(0),
            'v': instances['pred_densepose_v'][i].unsqueeze(0),
        }, boxes_xywh[i]) for i in range(len(boxes_xyxy))]
        return results, boxes_xywh


class MatrixVisualizer:
    def __init__(
            self, inplace=True, cmap=cv2.COLORMAP_PARULA, val_scale=1.0, alpha=0.7,
            interp_method_matrix=cv2.INTER_LINEAR, interp_method_mask=cv2.INTER_NEAREST,
    ):
        self.inplace = inplace
        self.cmap = cmap
        self.val_scale = val_scale
        self.alpha = alpha
        self.interp_method_matrix = interp_method_matrix
        self.interp_method_mask = interp_method_mask

    def visualize(self, image_bgr, mask, matrix, bbox_xywh):
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

    def fill(self, image_bgr, val=0):
        image_bgr[:] = (cv2.applyColorMap(np.array(val, dtype=np.uint8), self.cmap).reshape((1, 1, 3)) * self.alpha +
                        image_bgr * (1.0 - self.alpha))

    def _resize(self, mask, matrix, w, h):
        if (w != mask.shape[1]) or (h != mask.shape[0]):
            mask = cv2.resize(mask, (w, h), self.interp_method_mask)
        if (w != matrix.shape[1]) or (h != matrix.shape[0]):
            matrix = cv2.resize(matrix, (w, h), self.interp_method_matrix)
        return mask, matrix


def _extract_i_from_iuvarr(iuv_arr):
    return iuv_arr[0, :, :]


class DensePoseResultsFineSegmentationVisualizer:
    def __init__(self, inplace=True, cmap=cv2.COLORMAP_VIRIDIS, alpha=0.7, val_scale=255 / 24, keep_bg=True):
        self.mask_visualizer = MatrixVisualizer(inplace=inplace, cmap=cmap, val_scale=val_scale, alpha=alpha)
        self.keep_bg = keep_bg

    def visualize_iuv_arr(self, image_bgr, iuv_arr: np.ndarray, bbox_xywh) -> None:
        matrix = _extract_i_from_iuvarr(iuv_arr)
        segm = _extract_i_from_iuvarr(iuv_arr)
        mask = np.zeros(matrix.shape, dtype=np.uint8)
        mask[segm > 0] = 1
        self.mask_visualizer.visualize(image_bgr, mask, matrix, bbox_xywh)

    def visualize(self, image_bgr: np.ndarray, results_and_boxes_xywh) -> np.ndarray:
        densepose_result, boxes_xywh = results_and_boxes_xywh
        if densepose_result is None or boxes_xywh is None:
            return image_bgr

        if not self.keep_bg:
            self.mask_visualizer.fill(image_bgr, 0)
        boxes_xywh = boxes_xywh.cpu().numpy()
        for i, result in enumerate(densepose_result):
            iuv_array = torch.cat((result['labels'][None].type(torch.float32), result['uv'] * 255.0)).byte()
            self.visualize_iuv_arr(image_bgr, iuv_array.cpu().numpy(), boxes_xywh[i])
        return image_bgr


class End2EndVisualizer:
    def __init__(self, alpha=0.7, cmap=cv2.COLORMAP_VIRIDIS, keep_bg=True):
        self.extractor = DensePoseResultExtractor()
        self.visualizer = DensePoseResultsFineSegmentationVisualizer(alpha=alpha, cmap=cmap, keep_bg=keep_bg)

    def visualize(self, image_bgr: np.ndarray, instances) -> np.ndarray:
        data = self.extractor(instances)
        return self.visualizer.visualize(image_bgr, data)
