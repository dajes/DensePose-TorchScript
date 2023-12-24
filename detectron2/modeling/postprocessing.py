# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple, Dict

import torch
from torch.nn import functional as F

from detectron2.structures import scale_boxes, clip_boxes, nonempty_boxes


# perhaps should rename to "resize_instance"
def detector_postprocess(
        results: Dict[str, torch.Tensor], output_height: int, output_width: int, padding: Tuple[int, int, int, int]
) -> Dict[str, torch.Tensor]:
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results: the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.
    Returns:
        the resized output from the model, based on the output resolution
    """
    if isinstance(output_width, torch.Tensor):
        # This shape might (but not necessarily) be tensors during tracing.
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = torch.tensor((output_height, output_width), dtype=torch.int64)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / (results['image_size'][0] - padding[0] - padding[1]),
        output_height_tmp / (results['image_size'][1] - padding[2] - padding[3]),
    )

    output_boxes = results['pred_boxes']
    scale_boxes(output_boxes, scale_x, scale_y)

    keep = nonempty_boxes(output_boxes)
    return {
        'image_size': new_size,
        'pred_boxes': clip_boxes(output_boxes[keep], new_size),
        'scores': results['scores'][keep],
        'pred_classes': results['pred_classes'][keep],
        'pred_densepose_coarse_segm': results['pred_densepose_coarse_segm'][keep],
        'pred_densepose_fine_segm': results['pred_densepose_fine_segm'][keep],
        'pred_densepose_u': results['pred_densepose_u'][keep],
        'pred_densepose_v': results['pred_densepose_v'][keep],
    }


def sem_seg_postprocess(result, img_size, output_height, output_width):
    """
    Return semantic segmentation predictions in the original resolution.

    The input images are often resized when entering semantic segmentor. Moreover, in same
    cases, they also padded inside segmentor to be divisible by maximum network stride.
    As a result, we often need the predictions of the segmentor in a different
    resolution from its inputs.

    Args:
        result (Tensor): semantic segmentation prediction logits. A tensor of shape (C, H, W),
            where C is the number of classes, and H, W are the height and width of the prediction.
        img_size (tuple): image size that segmentor is taking as input.
        output_height, output_width: the desired output resolution.

    Returns:
        semantic segmentation prediction (Tensor): A tensor of the shape
            (C, output_height, output_width) that contains per-pixel soft predictions.
    """
    result = result[:, : img_size[0], : img_size[1]].expand(1, -1, -1, -1)
    result = F.interpolate(
        result, size=(output_height, output_width), mode="bilinear", align_corners=False
    )[0]
    return result
