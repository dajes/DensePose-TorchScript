import argparse
import os

import torch

from densepose import add_densepose_config
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor


def main():
    parser = argparse.ArgumentParser(description='Export DensePose model to TorchScript module')
    parser.add_argument("cfg", type=str, help="Config file")
    parser.add_argument("model", type=str, help="Model file")
    parser.add_argument("--min_score", default=0.3, type=float,
                        help="Minimum detection score to visualize")
    parser.add_argument("--nms_thresh", metavar="<threshold>", default=None, type=float,
                        help="NMS threshold")
    parser.add_argument("--fp16", action="store_true", help="Convert model to FP16")
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
    predictor = torch.jit.script(predictor)
    if args.fp16:
        predictor = predictor.half()
    os.makedirs("exported", exist_ok=True)
    model_path = f"exported/{args.cfg.split('/')[-1].split('.')[0]}_fp{['32', '16'][args.fp16]}.pt"
    torch.jit.save(predictor, model_path)
    print(f"Model saved to {model_path}")
    return


if __name__ == "__main__":
    main()
