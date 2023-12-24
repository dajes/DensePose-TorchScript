import argparse
import os

import cv2
import torch
import torchvision  # noqa: F401

from visualizer import End2EndVisualizer

parser = argparse.ArgumentParser(description='Export DensePose model to TorchScript module')
parser.add_argument("model", type=str, help="Model file")
parser.add_argument("input", type=str, help="Input data")
parser.add_argument("--cpu", action="store_true", help="Only use CPU")
args = parser.parse_args()
file_list = [args.input]
img = cv2.imread(args.input)
tensor = torch.from_numpy(img)

visualizer = End2EndVisualizer(alpha=1.0, keep_bg=False)
predictor = torch.jit.load(args.model)

if torch.cuda.is_available() and not args.cpu:
    tensor = tensor.cuda()
    predictor = predictor.cuda()
else:
    predictor = predictor.float()

outputs = predictor(tensor)
image_vis = visualizer.visualize(img, outputs)

save_path = "_pred".join(os.path.splitext(args.input))
cv2.imwrite(save_path, image_vis)
print(f"Image saved to {save_path}")
