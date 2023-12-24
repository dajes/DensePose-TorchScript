import argparse
import os
from itertools import count

import cv2
import torch
import torchvision  # noqa: F401

from visualizer import End2EndVisualizer

parser = argparse.ArgumentParser(description='Export DensePose model to TorchScript module')
parser.add_argument("model", type=str, help="Model file")
parser.add_argument("input", type=str, help="Input data")
parser.add_argument("--cpu", action="store_true", help="Only use CPU")
parser.add_argument("--fp32", action="store_true", help="Only use FP32")
args = parser.parse_args()
visualizer = End2EndVisualizer(alpha=.7, keep_bg=False)
predictor = torch.jit.load(args.model).eval()

if torch.cuda.is_available() and not args.cpu:
    device = torch.device("cuda")
    predictor = predictor.cuda()
    if args.fp32:
        predictor = predictor.float()
    else:
        predictor = predictor.half()
else:
    device = torch.device("cpu")
    predictor = predictor.float()

save_path = "_pred".join(os.path.splitext(args.input))
if os.path.splitext(args.input)[1].lower() in [".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"]:
    img = cv2.imread(args.input)
    tensor = torch.from_numpy(img)

    outputs = predictor(tensor)
    image_vis = visualizer.visualize(img, outputs)

    cv2.imwrite(save_path, image_vis)
    print(f"Image saved to {save_path}")
else:
    cap = cv2.VideoCapture(args.input)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    try:
        for i in count():
            ret, frame = cap.read()
            if not ret:
                break
            tensor = torch.from_numpy(frame)
            outputs = predictor(tensor)
            image_vis = visualizer.visualize(frame, outputs)
            if writer is None:
                writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (image_vis.shape[1], image_vis.shape[0]))
            writer.write(image_vis)
            print(f"Frame {i + 1}/{n_frames} processed", end="\r")
    except KeyboardInterrupt:
        pass
    if writer is not None:
        writer.release()
        print(f"Video saved to {save_path}")
    else:
        print("No frames processed")
