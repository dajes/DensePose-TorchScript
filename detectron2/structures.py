# Copyright (c) Facebook, Inc. and its affiliates.
from __future__ import division

import itertools
import math
import warnings
from enum import IntEnum, unique
from typing import Any, Iterator
from typing import Dict
from typing import List, Tuple, Union
from typing import Optional

import numpy as np
import torch

from detectron2.layers.roi_align import ROIAlign


class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size.
    The original sizes of each image is stored in `image_sizes`.

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w).
            During tracing, it becomes list[Tensor] instead.
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @staticmethod
    def from_tensors(
            tensors: List[torch.Tensor],
            size_divisibility: int = 0,
            pad_value: float = 0.0,
            padding_constraints: Optional[Dict[str, int]] = None,
    ) -> "ImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensor`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
                the common height and width is divisible by `size_divisibility`.
                This depends on the model and many models need a divisibility of 32.
            pad_value (float): value to pad.
            padding_constraints (optional[Dict]): If given, it would follow the format as
                {"size_divisibility": int, "square_size": int}, where `size_divisibility` will
                overwrite the above one if presented and `square_size` indicates the
                square padding size if `square_size` > 0.
        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[:-2] == tensors[0].shape[:-2], t.shape

        image_sizes = [(im.shape[-2], im.shape[-1]) for im in tensors]
        image_sizes_tensor = [torch.as_tensor(x) for x in image_sizes]
        max_size = torch.stack(image_sizes_tensor).max(0).values

        if padding_constraints is not None:
            square_size = padding_constraints.get("square_size", 0)
            if square_size > 0:
                # pad to square.
                max_size[0] = max_size[1] = square_size
            if "size_divisibility" in padding_constraints:
                size_divisibility = padding_constraints["size_divisibility"]
        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = (max_size + (stride - 1)).div(stride, rounding_mode="floor") * stride

        # handle weirdness of scripting and tracing ...
        if torch.jit.is_scripting():
            max_size: List[int] = max_size.to(dtype=torch.long).tolist()
        else:
            if torch.jit.is_tracing():
                image_sizes = image_sizes_tensor

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            # TODO: check whether it's faster for multiple images as well
            image_size = image_sizes[0]
            padding_size = [0, max_size[-1] - image_size[1], 0, max_size[-2] - image_size[0]]
            batched_imgs = F.pad(tensors[0], padding_size, value=pad_value).unsqueeze_(0)
        else:
            # max_size can be a tensor in tracing mode, therefore convert to list
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            device = (
                None if torch.jit.is_scripting() else ("cpu" if torch.jit.is_tracing() else None)
            )
            batched_imgs = tensors[0].new_full(batch_shape, pad_value, device=device)
            batched_imgs = batched_imgs.to(tensors[0].device)
            for i, img in enumerate(tensors):
                # Use `batched_imgs` directly instead of `img, pad_img = zip(tensors, batched_imgs)`
                # Tracing mode cannot capture `copy_()` of temporary locals
                batched_imgs[i, ..., : img.shape[-2], : img.shape[-1]].copy_(img)

        return ImageList(batched_imgs.contiguous(), image_sizes)


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        with warnings.catch_warnings(record=True):
            data_len = len(value)
        if len(self._fields):
            assert (
                    len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        if not isinstance(image_size, torch.Tensor):  # could be a tensor in tracing
            for i in instance_lists[1:]:
                assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__


def polygon_area(x, y):
    # Using the shoelace formula
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class Boxes:
    """
    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32, device=torch.device("cpu"))
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item) -> "Boxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
                (self.tensor[..., 0] >= -boundary_threshold)
                & (self.tensor[..., 1] >= -boundary_threshold)
                & (self.tensor[..., 2] < width + boundary_threshold)
                & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :2] + self.tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from self.tensor


class BitMasks:
    """
    This class stores the segmentation masks for all objects in one image, in
    the form of bitmaps.

    Attributes:
        tensor: bool Tensor of N,H,W, representing N instances in the image.
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor: bool Tensor of N,H,W, representing N instances in the image.
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(torch.bool)
        else:
            tensor = torch.as_tensor(tensor, dtype=torch.bool, device=torch.device("cpu"))
        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "BitMasks":
        return BitMasks(self.tensor.to(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @torch.jit.unused
    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "BitMasks":
        """
        Returns:
            BitMasks: Create a new :class:`BitMasks` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[3]`: return a `BitMasks` which contains only one mask.
        2. `new_masks = masks[2:10]`: return a slice of masks.
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return BitMasks(self.tensor[item].unsqueeze(0))
        m = self.tensor[item]
        assert m.dim() == 3, "Indexing on BitMasks with {} returns a tensor with shape {}!".format(
            item, m.shape
        )
        return BitMasks(m)

    @torch.jit.unused
    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    @torch.jit.unused
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        return self.tensor.flatten(1).any(dim=1)

    @staticmethod
    def from_roi_masks(roi_masks: "ROIMasks", height: int, width: int) -> "BitMasks":
        """
        Args:
            roi_masks:
            height, width (int):
        """
        return roi_masks.to_bitmasks(height, width)

    def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
        """
        Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
        This can be used to prepare training targets for Mask R-CNN.
        It has less reconstruction error compared to rasterization with polygons.
        However we observe no difference in accuracy,
        but BitMasks requires more memory to store all the masks.

        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each mask
            mask_size (int): the size of the rasterized mask.

        Returns:
            Tensor:
                A bool tensor of shape (N, mask_size, mask_size), where
                N is the number of predicted boxes for this image.
        """
        assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))
        device = self.tensor.device

        batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
        rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

        bit_masks = self.tensor.to(dtype=torch.float32)
        rois = rois.to(device=device)
        output = (
            ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
            .forward(bit_masks[:, None, :, :], rois)
            .squeeze(1)
        )
        output = output >= 0.5
        return output

    def get_bounding_boxes(self) -> Boxes:
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
        """
        boxes = torch.zeros(self.tensor.shape[0], 4, dtype=torch.float32)
        x_any = torch.any(self.tensor, dim=1)
        y_any = torch.any(self.tensor, dim=2)
        for idx in range(self.tensor.shape[0]):
            x = torch.where(x_any[idx, :])[0]
            y = torch.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = torch.as_tensor(
                    [x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=torch.float32
                )
        return Boxes(boxes)

    @staticmethod
    def cat(bitmasks_list: List["BitMasks"]) -> "BitMasks":
        """
        Concatenates a list of BitMasks into a single BitMasks

        Arguments:
            bitmasks_list (list[BitMasks])

        Returns:
            BitMasks: the concatenated BitMasks
        """
        assert isinstance(bitmasks_list, (list, tuple))
        assert len(bitmasks_list) > 0
        assert all(isinstance(bitmask, BitMasks) for bitmask in bitmasks_list)

        cat_bitmasks = type(bitmasks_list[0])(torch.cat([bm.tensor for bm in bitmasks_list], dim=0))
        return cat_bitmasks


class PolygonMasks:
    """
    This class stores the segmentation masks for all objects in one image, in the form of polygons.

    Attributes:
        polygons: list[list[ndarray]]. Each ndarray is a float64 vector representing a polygon.
    """

    def __init__(self, polygons: List[List[Union[torch.Tensor, np.ndarray]]]):
        """
        Arguments:
            polygons (list[list[np.ndarray]]): The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                instance, and the third level to the polygon coordinates.
                The third level array should have the format of
                [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
        """
        if not isinstance(polygons, list):
            raise ValueError(
                "Cannot create PolygonMasks: Expect a list of list of polygons per image. "
                "Got '{}' instead.".format(type(polygons))
            )

        def _make_array(t: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
            # Use float64 for higher precision, because why not?
            # Always put polygons on CPU (self.to is a no-op) since they
            # are supposed to be small tensors.
            # May need to change this assumption if GPU placement becomes useful
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()
            return np.asarray(t).astype("float64")

        def process_polygons(
                polygons_per_instance: List[Union[torch.Tensor, np.ndarray]]
        ) -> List[np.ndarray]:
            if not isinstance(polygons_per_instance, list):
                raise ValueError(
                    "Cannot create polygons: Expect a list of polygons per instance. "
                    "Got '{}' instead.".format(type(polygons_per_instance))
                )
            # transform each polygon to a numpy array
            polygons_per_instance = [_make_array(p) for p in polygons_per_instance]
            for polygon in polygons_per_instance:
                if len(polygon) % 2 != 0 or len(polygon) < 6:
                    raise ValueError(f"Cannot create a polygon from {len(polygon)} coordinates.")
            return polygons_per_instance

        self.polygons: List[List[np.ndarray]] = [
            process_polygons(polygons_per_instance) for polygons_per_instance in polygons
        ]

    def to(self, *args: Any, **kwargs: Any) -> "PolygonMasks":
        return self

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def get_bounding_boxes(self) -> Boxes:
        """
        Returns:
            Boxes: tight bounding boxes around polygon masks.
        """
        boxes = torch.zeros(len(self.polygons), 4, dtype=torch.float32)
        for idx, polygons_per_instance in enumerate(self.polygons):
            minxy = torch.as_tensor([float("inf"), float("inf")], dtype=torch.float32)
            maxxy = torch.zeros(2, dtype=torch.float32)
            for polygon in polygons_per_instance:
                coords = torch.from_numpy(polygon).view(-1, 2).to(dtype=torch.float32)
                minxy = torch.min(minxy, torch.min(coords, dim=0).values)
                maxxy = torch.max(maxxy, torch.max(coords, dim=0).values)
            boxes[idx, :2] = minxy
            boxes[idx, 2:] = maxxy
        return Boxes(boxes)

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor:
                a BoolTensor which represents whether each mask is empty (False) or not (True).
        """
        keep = [1 if len(polygon) > 0 else 0 for polygon in self.polygons]
        return torch.from_numpy(np.asarray(keep, dtype=bool))

    def __getitem__(self, item: Union[int, slice, List[int], torch.BoolTensor]) -> "PolygonMasks":
        """
        Support indexing over the instances and return a `PolygonMasks` object.
        `item` can be:

        1. An integer. It will return an object with only one instance.
        2. A slice. It will return an object with the selected instances.
        3. A list[int]. It will return an object with the selected instances,
           correpsonding to the indices in the list.
        4. A vector mask of type BoolTensor, whose length is num_instances.
           It will return an object with the instances whose mask is nonzero.
        """
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        elif isinstance(item, list):
            selected_polygons = [self.polygons[i] for i in item]
        elif isinstance(item, torch.Tensor):
            # Polygons is a list, so we have to move the indices back to CPU.
            if item.dtype == torch.bool:
                assert item.dim() == 1, item.shape
                item = item.nonzero().squeeze(1).cpu().numpy().tolist()
            elif item.dtype in [torch.int32, torch.int64]:
                item = item.cpu().numpy().tolist()
            else:
                raise ValueError("Unsupported tensor dtype={} for indexing!".format(item.dtype))
            selected_polygons = [self.polygons[i] for i in item]
        return PolygonMasks(selected_polygons)

    def __iter__(self) -> Iterator[List[np.ndarray]]:
        """
        Yields:
            list[ndarray]: the polygons for one instance.
            Each Tensor is a float64 vector representing a polygon.
        """
        return iter(self.polygons)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.polygons))
        return s

    def __len__(self) -> int:
        return len(self.polygons)

    def area(self):
        """
        Computes area of the mask.
        Only works with Polygons, using the shoelace formula:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

        Returns:
            Tensor: a vector, area for each instance
        """

        area = []
        for polygons_per_instance in self.polygons:
            area_per_instance = 0
            for p in polygons_per_instance:
                area_per_instance += polygon_area(p[0::2], p[1::2])
            area.append(area_per_instance)

        return torch.tensor(area)

    @staticmethod
    def cat(polymasks_list: List["PolygonMasks"]) -> "PolygonMasks":
        """
        Concatenates a list of PolygonMasks into a single PolygonMasks

        Arguments:
            polymasks_list (list[PolygonMasks])

        Returns:
            PolygonMasks: the concatenated PolygonMasks
        """
        assert isinstance(polymasks_list, (list, tuple))
        assert len(polymasks_list) > 0
        assert all(isinstance(polymask, PolygonMasks) for polymask in polymasks_list)

        cat_polymasks = type(polymasks_list[0])(
            list(itertools.chain.from_iterable(pm.polygons for pm in polymasks_list))
        )
        return cat_polymasks


class ROIMasks:
    """
    Represent masks by N smaller masks defined in some ROIs. Once ROI boxes are given,
    full-image bitmask can be obtained by "pasting" the mask on the region defined
    by the corresponding ROI box.
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor: (N, M, M) mask tensor that defines the mask within each ROI.
        """
        if tensor.dim() != 3:
            raise ValueError("ROIMasks must take a masks of 3 dimension.")
        self.tensor = tensor

    def to(self, device: torch.device) -> "ROIMasks":
        return ROIMasks(self.tensor.to(device))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, item) -> "ROIMasks":
        """
        Returns:
            ROIMasks: Create a new :class:`ROIMasks` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[2:10]`: return a slice of masks.
        2. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        t = self.tensor[item]
        if t.dim() != 3:
            raise ValueError(
                f"Indexing on ROIMasks with {item} returns a tensor with shape {t.shape}!"
            )
        return ROIMasks(t)

    @torch.jit.unused
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s


_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


@unique
class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYXY_ABS = 0
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 1
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """
    XYXY_REL = 2
    """
    Not yet supported!
    (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
    """
    XYWH_REL = 3
    """
    Not yet supported!
    (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """
    XYWHA_ABS = 4
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """

    @staticmethod
    def convert(box: _RawBoxType, from_mode: "BoxMode", to_mode: "BoxMode") -> _RawBoxType:
        """
        Args:
            box: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
            from_mode, to_mode (BoxMode)

        Returns:
            The converted box of the same type.
        """
        if from_mode == to_mode:
            return box

        original_type = type(box)
        is_numpy = isinstance(box, np.ndarray)
        single_box = isinstance(box, (list, tuple))
        if single_box:
            assert len(box) == 4 or len(box) == 5, (
                "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
                " where k == 4 or 5"
            )
            arr = torch.tensor(box)[None, :]
        else:
            # avoid modifying the input box
            if is_numpy:
                arr = torch.from_numpy(np.asarray(box)).clone()
            else:
                arr = box.clone()

        assert to_mode not in [BoxMode.XYXY_REL, BoxMode.XYWH_REL] and from_mode not in [
            BoxMode.XYXY_REL,
            BoxMode.XYWH_REL,
        ], "Relative mode not yet supported!"

        if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
            assert (
                    arr.shape[-1] == 5
            ), "The last dimension of input shape must be 5 for XYWHA format"
            original_dtype = arr.dtype
            arr = arr.double()

            w = arr[:, 2]
            h = arr[:, 3]
            a = arr[:, 4]
            c = torch.abs(torch.cos(a * math.pi / 180.0))
            s = torch.abs(torch.sin(a * math.pi / 180.0))
            # This basically computes the horizontal bounding rectangle of the rotated box
            new_w = c * w + s * h
            new_h = c * h + s * w

            # convert center to top-left corner
            arr[:, 0] -= new_w / 2.0
            arr[:, 1] -= new_h / 2.0
            # bottom-right corner
            arr[:, 2] = arr[:, 0] + new_w
            arr[:, 3] = arr[:, 1] + new_h

            arr = arr[:, :4].to(dtype=original_dtype)
        elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
            original_dtype = arr.dtype
            arr = arr.double()
            arr[:, 0] += arr[:, 2] / 2.0
            arr[:, 1] += arr[:, 3] / 2.0
            angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
            arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
        else:
            if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
                arr[:, 2] += arr[:, 0]
                arr[:, 3] += arr[:, 1]
            elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
                arr[:, 2] -= arr[:, 0]
                arr[:, 3] -= arr[:, 1]
            else:
                raise NotImplementedError(
                    "Conversion from BoxMode {} to {} is not supported yet".format(
                        from_mode, to_mode
                    )
                )

        if single_box:
            return original_type(arr.flatten().tolist())
        if is_numpy:
            return arr.numpy()
        else:
            return arr


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: intersection, sized [N,M].
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Similar to :func:`pariwise_iou` but compute the IoA (intersection over boxes2 area).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoA, sized [N,M].
    """
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    ioa = torch.where(
        inter > 0, inter / area2, torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return ioa


def pairwise_point_box_distance(points: torch.Tensor, boxes: Boxes):
    """
    Pairwise distance between N points and M boxes. The distance between a
    point and a box is represented by the distance from the point to 4 edges
    of the box. Distances are all positive when the point is inside the box.

    Args:
        points: Nx2 coordinates. Each row is (x, y)
        boxes: M boxes

    Returns:
        Tensor: distances of size (N, M, 4). The 4 values are distances from
            the point to the left, top, right, bottom of the box.
    """
    x, y = points.unsqueeze(dim=2).unbind(dim=1)  # (N, 1)
    x0, y0, x1, y1 = boxes.tensor.unsqueeze(dim=0).unbind(dim=2)  # (1, M)
    return torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)


def matched_pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes that have the same number of boxes.
    Similar to :func:`pairwise_iou`, but computes only diagonal elements of the matrix.

    Args:
        boxes1 (Boxes): bounding boxes, sized [N,4].
        boxes2 (Boxes): same length as boxes1
    Returns:
        Tensor: iou, sized [N].
    """
    assert len(boxes1) == len(
        boxes2
    ), "boxlists should have the same" "number of entries, got {}, {}".format(
        len(boxes1), len(boxes2)
    )
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou


# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import Any, List, Tuple, Union
import torch
from torch.nn import functional as F


class Keypoints:
    """
    Stores keypoint **annotation** data. GT Instances have a `gt_keypoints` property
    containing the x,y location and visibility flag of each keypoint. This tensor has shape
    (N, K, 3) where N is the number of instances and K is the number of keypoints per instance.

    The visibility flag follows the COCO format and must be one of three integers:

    * v=0: not labeled (in which case x=y=0)
    * v=1: labeled but not visible
    * v=2: labeled and visible
    """

    def __init__(self, keypoints: Union[torch.Tensor, np.ndarray, List[List[float]]]):
        """
        Arguments:
            keypoints: A Tensor, numpy array, or list of the x, y, and visibility of each keypoint.
                The shape should be (N, K, 3) where N is the number of
                instances, and K is the number of keypoints per instance.
        """
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device("cpu")
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        assert keypoints.dim() == 3 and keypoints.shape[2] == 3, keypoints.shape
        self.tensor = keypoints

    def __len__(self) -> int:
        return self.tensor.size(0)

    def to(self, *args: Any, **kwargs: Any) -> "Keypoints":
        return type(self)(self.tensor.to(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def to_heatmap(self, boxes: torch.Tensor, heatmap_size: int) -> torch.Tensor:
        """
        Convert keypoint annotations to a heatmap of one-hot labels for training,
        as described in :paper:`Mask R-CNN`.

        Arguments:
            boxes: Nx4 tensor, the boxes to draw the keypoints to

        Returns:
            heatmaps:
                A tensor of shape (N, K), each element is integer spatial label
                in the range [0, heatmap_size**2 - 1] for each keypoint in the input.
            valid:
                A tensor of shape (N, K) containing whether each keypoint is in the roi or not.
        """
        return _keypoints_to_heatmap(self.tensor, boxes, heatmap_size)

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Keypoints":
        """
        Create a new `Keypoints` by indexing on this `Keypoints`.

        The following usage are allowed:

        1. `new_kpts = kpts[3]`: return a `Keypoints` which contains only one instance.
        2. `new_kpts = kpts[2:10]`: return a slice of key points.
        3. `new_kpts = kpts[vector]`, where vector is a torch.ByteTensor
           with `length = len(kpts)`. Nonzero elements in the vector will be selected.

        Note that the returned Keypoints might share storage with this Keypoints,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Keypoints([self.tensor[item]])
        return Keypoints(self.tensor[item])

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    @staticmethod
    def cat(keypoints_list: List["Keypoints"]) -> "Keypoints":
        """
        Concatenates a list of Keypoints into a single Keypoints

        Arguments:
            keypoints_list (list[Keypoints])

        Returns:
            Keypoints: the concatenated Keypoints
        """
        assert isinstance(keypoints_list, (list, tuple))
        assert len(keypoints_list) > 0
        assert all(isinstance(keypoints, Keypoints) for keypoints in keypoints_list)

        cat_kpts = type(keypoints_list[0])(
            torch.cat([kpts.tensor for kpts in keypoints_list], dim=0)
        )
        return cat_kpts


# TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
def _keypoints_to_heatmap(
        keypoints: torch.Tensor, rois: torch.Tensor, heatmap_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode keypoint locations into a target heatmap for use in SoftmaxWithLoss across space.

    Maps keypoints from the half-open interval [x1, x2) on continuous image coordinates to the
    closed interval [0, heatmap_size - 1] on discrete image coordinates. We use the
    continuous-discrete conversion from Heckbert 1990 ("What is the coordinate of a pixel?"):
    d = floor(c) and c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.

    Arguments:
        keypoints: tensor of keypoint locations in of shape (N, K, 3).
        rois: Nx4 tensor of rois in xyxy format
        heatmap_size: integer side length of square heatmap.

    Returns:
        heatmaps: A tensor of shape (N, K) containing an integer spatial label
            in the range [0, heatmap_size**2 - 1] for each keypoint in the input.
        valid: A tensor of shape (N, K) containing whether each keypoint is in
            the roi or not.
    """

    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid


@torch.jit.script_if_tracing
def heatmaps_to_keypoints(maps: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
    """
    Extract predicted keypoint locations from heatmaps.

    Args:
        maps (Tensor): (#ROIs, #keypoints, POOL_H, POOL_W). The predicted heatmap of logits for
            each ROI and each keypoint.
        rois (Tensor): (#ROIs, 4). The box of each ROI.

    Returns:
        Tensor of shape (#ROIs, #keypoints, 4) with the last dimension corresponding to
        (x, y, logit, score) for each keypoint.

    When converting discrete pixel indices in an NxN image to a continuous keypoint coordinate,
    we maintain consistency with :meth:`Keypoints.to_heatmap` by using the conversion from
    Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.
    """

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_rois, num_keypoints = maps.shape[:2]
    xy_preds = maps.new_zeros(rois.shape[0], num_keypoints, 4)

    width_corrections = widths / widths_ceil
    height_corrections = heights / heights_ceil

    keypoints_idx = torch.arange(num_keypoints, device=maps.device)

    for i in range(num_rois):
        outsize = (int(heights_ceil[i]), int(widths_ceil[i]))
        roi_map = F.interpolate(maps[[i]], size=outsize, mode="bicubic", align_corners=False)

        # Although semantically equivalent, `reshape` is used instead of `squeeze` due
        # to limitation during ONNX export of `squeeze` in scripting mode
        roi_map = roi_map.reshape(roi_map.shape[1:])  # keypoints x H x W

        # softmax over the spatial region
        max_score, _ = roi_map.view(num_keypoints, -1).max(1)
        max_score = max_score.view(num_keypoints, 1, 1)
        tmp_full_resolution = (roi_map - max_score).exp_()
        tmp_pool_resolution = (maps[i] - max_score).exp_()
        # Produce scores over the region H x W, but normalize with POOL_H x POOL_W,
        # so that the scores of objects of different absolute sizes will be more comparable
        roi_map_scores = tmp_full_resolution / tmp_pool_resolution.sum((1, 2), keepdim=True)

        w = roi_map.shape[2]
        pos = roi_map.view(num_keypoints, -1).argmax(1)

        x_int = pos % w
        y_int = (pos - x_int) // w

        assert (
                roi_map_scores[keypoints_idx, y_int, x_int]
                == roi_map_scores.view(num_keypoints, -1).max(1)[0]
        ).all()

        x = (x_int.float() + 0.5) * width_corrections[i]
        y = (y_int.float() + 0.5) * height_corrections[i]

        xy_preds[i, :, 0] = x + offset_x[i]
        xy_preds[i, :, 1] = y + offset_y[i]
        xy_preds[i, :, 2] = roi_map[keypoints_idx, y_int, x_int]
        xy_preds[i, :, 3] = roi_map_scores[keypoints_idx, y_int, x_int]

    return xy_preds
