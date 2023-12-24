import argparse

import cv2
import numpy as np
import torch

from densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor


def main():
    parser = argparse.ArgumentParser(description='Export DensePose model to TorchScript module')
    parser.add_argument("cfg", type=str, help="Config file")
    parser.add_argument("model", type=str, help="Model file")
    parser.add_argument("input", type=str, help="Input data")
    parser.add_argument("--min_score", default=0.8, type=float,
                        help="Minimum detection score to visualize")
    parser.add_argument("--nms_thresh", metavar="<threshold>", default=None, type=float,
                        help="NMS threshold")
    args = parser.parse_args()
    opts = []
    cfg = get_cfg()
    opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
    opts.append(str(args.min_score))
    if args.nms_thresh is not None:
        opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
        opts.append(str(args.nms_thresh))
    add_densepose_config(cfg)
    cfg.merge_from_file(args.cfg)
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = args.model
    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    file_list = [args.input]
    if len(file_list) == 0:
        return
    from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer
    from densepose.vis.extractor import DensePoseResultExtractor
    visualizer = DensePoseResultsFineSegmentationVisualizer(cfg)
    extractor = DensePoseResultExtractor()
    img = cv2.imread(args.input)
    tensor = torch.from_numpy(img)
    outputs = predictor(tensor)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    data = extractor(outputs)
    image_vis = visualizer.visualize(image, data)
    predictor = torch.jit.script(predictor)
    outputs2 = predictor(tensor)
    data2 = extractor(outputs2)
    image_vis2 = visualizer.visualize(image, data2)
    diff = np.abs(image_vis.astype(np.int16) - image_vis2).astype(np.uint8)
    return diff


if __name__ == "__main__":
    main()
