from dataclasses import dataclass
from typing import Any, Tuple, Type, Dict
from typing import Optional

import torch
from torch.nn import functional as F

from detectron2.structures import Boxes


class BaseConverter:
    """
    Converter base class to be reused by various converters.
    Converter allows one to convert data from various source types to a particular
    destination type. Each source type needs to register its converter. The
    registration for each source type is valid for all descendants of that type.
    """

    @classmethod
    def register(cls, from_type: Type, converter: Any = None):
        """
        Registers a converter for the specified type.
        Can be used as a decorator (if converter is None), or called as a method.

        Args:
            from_type (type): type to register the converter for;
                all instances of this type will use the same converter
            converter (callable): converter to be registered for the given
                type; if None, this method is assumed to be a decorator for the converter
        """

        if converter is not None:
            cls._do_register(from_type, converter)

        def wrapper(converter: Any) -> Any:
            cls._do_register(from_type, converter)
            return converter

        return wrapper

    @classmethod
    def _do_register(cls, from_type: Type, converter: Any):
        cls.registry[from_type] = converter  # pyre-ignore[16]

    @classmethod
    def _lookup_converter(cls, from_type: Type) -> Any:
        """
        Perform recursive lookup for the given type
        to find registered converter. If a converter was found for some base
        class, it gets registered for this class to save on further lookups.

        Args:
            from_type: type for which to find a converter
        Return:
            callable or None - registered converter or None
                if no suitable entry was found in the registry
        """
        if from_type in cls.registry:  # pyre-ignore[16]
            return cls.registry[from_type]
        for base in from_type.__bases__:
            converter = cls._lookup_converter(base)
            if converter is not None:
                cls._do_register(from_type, converter)
                return converter
        return None

    @classmethod
    def convert(cls, instance: Any, *args, **kwargs):
        """
        Convert an instance to the destination type using some registered
        converter. Does recursive lookup for base classes, so there's no need
        for explicit registration for derived classes.

        Args:
            instance: source instance to convert to the destination type
        Return:
            An instance of the destination type obtained from the source instance
            Raises KeyError, if no suitable converter found
        """
        return densepose_chart_predictor_output_to_result_with_confidences(instance, *args, **kwargs)


@dataclass
class DensePoseChartResultWithConfidences:
    """
    We add confidence values to DensePoseChartResult
    Thus the results are represented by two tensors:
    - labels (tensor [H, W] of long): contains estimated label for each pixel of
        the detection bounding box of size (H, W)
    - uv (tensor [2, H, W] of float): contains estimated U and V coordinates
        for each pixel of the detection bounding box of size (H, W)
    Plus one [H, W] tensor of float for each confidence type
    """

    labels: torch.Tensor
    uv: torch.Tensor
    sigma_1: Optional[torch.Tensor] = None
    sigma_2: Optional[torch.Tensor] = None
    kappa_u: Optional[torch.Tensor] = None
    kappa_v: Optional[torch.Tensor] = None
    fine_segm_confidence: Optional[torch.Tensor] = None
    coarse_segm_confidence: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        """
        Transfers all tensors to the given device, except if their value is None
        """

        def to_device_if_tensor(var: Any):
            if isinstance(var, torch.Tensor):
                return var.to(device)
            return var

        return DensePoseChartResultWithConfidences(
            labels=self.labels.to(device),
            uv=self.uv.to(device),
            sigma_1=to_device_if_tensor(self.sigma_1),
            sigma_2=to_device_if_tensor(self.sigma_2),
            kappa_u=to_device_if_tensor(self.kappa_u),
            kappa_v=to_device_if_tensor(self.kappa_v),
            fine_segm_confidence=to_device_if_tensor(self.fine_segm_confidence),
            coarse_segm_confidence=to_device_if_tensor(self.coarse_segm_confidence),
        )


class ToChartResultConverterWithConfidences(BaseConverter):
    """
    Converts various DensePose predictor outputs to DensePose results.
    Each DensePose predictor output type has to register its convertion strategy.
    """

    registry = {}
    dst_type = DensePoseChartResultWithConfidences

    @classmethod
    # pyre-fixme[14]: `convert` overrides method defined in `BaseConverter`
    #  inconsistently.
    def convert(
            cls, predictor_outputs: Any, boxes: Boxes, *args, **kwargs
    ) -> DensePoseChartResultWithConfidences:
        """
        Convert DensePose predictor outputs to DensePoseResult with confidences
        using some registered converter. Does recursive lookup for base classes,
        so there's no need for explicit registration for derived classes.

        Args:
            densepose_predictor_outputs: DensePose predictor output with confidences
                to be converted to BitMasks
            boxes (Boxes): bounding boxes that correspond to the DensePose
                predictor outputs
        Return:
            An instance of DensePoseResult. If no suitable converter was found, raises KeyError
        """
        return super(ToChartResultConverterWithConfidences, cls).convert(
            predictor_outputs, boxes, *args, **kwargs
        )


def make_int_box(box: torch.Tensor):
    int_box = [0, 0, 0, 0]
    int_box[0], int_box[1], int_box[2], int_box[3] = tuple(box.long().tolist())
    return int_box[0], int_box[1], int_box[2], int_box[3]


def resample_fine_and_coarse_segm_tensors_to_bbox(
        fine_segm: torch.Tensor, coarse_segm: torch.Tensor, box_xywh_abs
):
    """
    Resample fine and coarse segmentation tensors to the given
    bounding box and derive labels for each pixel of the bounding box

    Args:
        fine_segm: float tensor of shape [1, C, Hout, Wout]
        coarse_segm: float tensor of shape [1, K, Hout, Wout]
        box_xywh_abs (tuple of 4 int): bounding box given by its upper-left
            corner coordinates, width (W) and height (H)
    Return:
        Labels for each pixel of the bounding box, a long tensor of size [1, H, W]
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    # coarse segmentation
    coarse_segm_bbox = F.interpolate(
        coarse_segm,
        (h, w),
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)
    # combined coarse and fine segmentation
    labels = (
            F.interpolate(fine_segm, (h, w), mode="bilinear", align_corners=False).argmax(dim=1)
            * (coarse_segm_bbox > 0).long()
    )
    return labels


def resample_fine_and_coarse_segm_to_bbox(predictor_output: Any, box_xywh_abs):
    """
    Resample fine and coarse segmentation outputs from a predictor to the given
    bounding box and derive labels for each pixel of the bounding box

    Args:
        predictor_output: DensePose predictor output that contains segmentation
            results to be resampled
        box_xywh_abs (tuple of 4 int): bounding box given by its upper-left
            corner coordinates, width (W) and height (H)
    Return:
        Labels for each pixel of the bounding box, a long tensor of size [1, H, W]
    """
    return resample_fine_and_coarse_segm_tensors_to_bbox(
        predictor_output['fine_segm'],
        predictor_output['coarse_segm'],
        box_xywh_abs,
    )


def resample_uv_tensors_to_bbox(
        u: torch.Tensor,
        v: torch.Tensor,
        labels: torch.Tensor,
        box_xywh_abs,
) -> torch.Tensor:
    """
    Resamples U and V coordinate estimates for the given bounding box

    Args:
        u (tensor [1, C, H, W] of float): U coordinates
        v (tensor [1, C, H, W] of float): V coordinates
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled U and V coordinates - a tensor [2, H, W] of float
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    u_bbox = F.interpolate(u, (h, w), mode="bilinear", align_corners=False)
    v_bbox = F.interpolate(v, (h, w), mode="bilinear", align_corners=False)
    uv = torch.zeros([2, h, w], dtype=torch.float32, device=u.device)
    for part_id in range(1, u_bbox.size(1)):
        uv[0][labels == part_id] = u_bbox[0, part_id][labels == part_id]
        uv[1][labels == part_id] = v_bbox[0, part_id][labels == part_id]
    return uv


def resample_uv_to_bbox(
        predictor_output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        box_xywh_abs,
) -> torch.Tensor:
    return resample_uv_tensors_to_bbox(
        predictor_output['u'],
        predictor_output['v'],
        labels,
        box_xywh_abs,
    )


def resample_confidences_to_bbox(
        predictor_output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        box_xywh_abs,
) -> Dict[str, torch.Tensor]:
    """
    Resamples confidences for the given bounding box

    Args:
        predictor_output: DensePose predictor
            output to be resampled
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled confidences - a dict of [H, W] tensors of float
    """

    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)

    confidence_names = [
        "sigma_1",
        "sigma_2",
        "kappa_u",
        "kappa_v",
        "fine_segm_confidence",
        "coarse_segm_confidence",
    ]
    confidence_results = {key: None for key in confidence_names}
    confidence_names = [key for key in confidence_names if predictor_output.get(key, None) is not None]
    confidence_base = torch.zeros([h, w], dtype=torch.float32, device=predictor_output['u'].device)

    # assign data from channels that correspond to the labels
    for key in confidence_names:
        resampled_confidence = F.interpolate(
            predictor_output[key],
            (h, w),
            mode="bilinear",
            align_corners=False,
        )
        result = confidence_base.clone()
        for part_id in range(1, predictor_output['u'].size(1)):
            if resampled_confidence.size(1) != predictor_output['u'].size(1):
                # confidence is not part-based, don't try to fill it part by part
                continue
            result[labels == part_id] = resampled_confidence[0, part_id][labels == part_id]

        if resampled_confidence.size(1) != predictor_output['u'].size(1):
            # confidence is not part-based, fill the data with the first channel
            # (targeted for segmentation confidences that have only 1 channel)
            result = resampled_confidence[0, 0]

        confidence_results[key] = result

    return confidence_results  # pyre-ignore[7]


def densepose_chart_predictor_output_to_result_with_confidences(
        predictor_output: Dict[str, torch.Tensor], boxes: Boxes
) -> DensePoseChartResultWithConfidences:
    """
    Convert densepose chart predictor outputs to results

    Args:
        predictor_output: DensePose predictor
            output with confidences to be converted to results, must contain only 1 output
        boxes (Boxes): bounding box that corresponds to the predictor output,
            must contain only 1 bounding box
    Return:
       DensePose chart-based result with confidences (DensePoseChartResultWithConfidences)
    """
    boxes_xyxy_abs = boxes.clone()
    boxes_xyxy_abs[:, 2:] -= boxes_xyxy_abs[:, :2]
    box_xywh = make_int_box(boxes_xyxy_abs[0])

    labels = resample_fine_and_coarse_segm_to_bbox(predictor_output, box_xywh).squeeze(0)
    uv = resample_uv_to_bbox(predictor_output, labels, box_xywh)
    confidences = resample_confidences_to_bbox(predictor_output, labels, box_xywh)
    return DensePoseChartResultWithConfidences(labels=labels, uv=uv, **confidences)


def extract_boxes_xywh_from_instances(instances):
    boxes_xywh = instances['pred_boxes'].clone()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]
    return boxes_xywh


class DensePoseResultExtractor:
    def __call__(self, instances) -> Tuple[Any, Optional[torch.Tensor]]:
        boxes_xyxy = instances['pred_boxes']
        boxes_xywh = extract_boxes_xywh_from_instances(instances)
        converter = ToChartResultConverterWithConfidences()
        results = [converter.convert({
            'coarse_segm': instances['pred_densepose_coarse_segm'][i].unsqueeze(0),
            'fine_segm': instances['pred_densepose_fine_segm'][i].unsqueeze(0),
            'u': instances['pred_densepose_u'][i].unsqueeze(0),
            'v': instances['pred_densepose_v'][i].unsqueeze(0),
        }, boxes_xyxy[[i]]) for i in range(len(boxes_xyxy))]
        return results, boxes_xywh
