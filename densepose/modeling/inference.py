# Copyright (c) Facebook, Inc. and its affiliates.
from typing import List, Dict

import torch


def densepose_inference(densepose_predictor_output: Dict[str, torch.Tensor],
                        detections: List[Dict[str, torch.Tensor]]) -> None:
    """
    Splits DensePose predictor outputs into chunks, each chunk corresponds to
    detections on one image. Predictor output chunks are stored in `pred_densepose`
    attribute of the corresponding object.

    Args:
        densepose_predictor_output: a dataclass instance (can be of different types,
            depending on predictor used for inference). Each field can be `None`
            (if the corresponding output was not inferred) or a tensor of size
            [N, ...], where N = N_1 + N_2 + .. + N_k is a total number of
            detections on all images, N_1 is the number of detections on image 1,
            N_2 is the number of detections on image 2, etc.
        detections: a list of objects of type `Instance`, k-th object corresponds
            to detections on k-th image.
    """
    k = 0
    for detection_i in detections:
        if densepose_predictor_output is None:
            continue
        n_i = detection_i['scores'].shape[0]
        for field, field_value in densepose_predictor_output.items():
            if isinstance(field_value, torch.Tensor):
                detection_i[f'pred_densepose_{field}'] = field_value[k: k + n_i]
        k += n_i
