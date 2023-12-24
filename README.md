<a href="https://savelife.in.ua/en/donate-en/"><img src="https://savelife.in.ua/wp-content/themes/savelife/assets/images/new-logo-en.svg" width=120px></a>
# Exportable DensePose inference using TorchScript

### This is unofficial inference implementation of [DensePose from detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)

The project is focused on creating simple and TorchScript compilable inference interface for the original pretrained
models to free them from the heavy dependency on the detectron2 framework.

#### Only inference is supported, no training. Also no confidence estimation or bootstapping pipelines were implemented.

# Quickstart
To run already exported model (which you might find in the 
[Releases](https://github.com/dajes/DensePose-TorchScript/releases) section) you only need PyTorch and OpenCV 
(for image reading):
    
```
pip install torch torchvision opencv-python
```

Then you can run the model using the small example script:

```
python run.py <model.pt> <input.[jpg|png|mp4|avi]>
```
This will run the model and save the result in the same directory as the input.


## Exporting a model by yourself

To export a model you need to have a model checkpoint and a config file. You can find them in the table below

```
python export.py <config> <model> [--fp16]
```

If --fp16 is specified, the model will be exported in fp16 mode. This will reduce the model size at the cost of
precision.

Example of exporting an R_50_FPN_s1x_legacy model into fp16 format model:

```
python export.py configs/densepose_rcnn_R_50_FPN_s1x_legacy.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x_legacy/164832157/model_final_d366fa.pkl --fp16
```

### License

All models available for download are licensed under the
[Creative Commons Attribution-ShareAlike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/)

### Legacy Models

Baselines trained using schedules from [Güler et al, 2018](https://arxiv.org/pdf/1802.00434.pdf)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">segm<br/>AP</th>
<th valign="bottom">dp. AP<br/>GPS</th>
<th valign="bottom">dp. AP<br/>GPSm</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_rcnn_R_50_FPN_s1x_legacy -->
<tr><td align="left"><a href="../configs/densepose_rcnn_R_50_FPN_s1x_legacy.yaml">R_50_FPN_s1x_legacy</a></td>
<td align="center">s1x</td>
<td align="center">0.307</td>
<td align="center">0.051</td>
<td align="center">3.2</td>
<td align="center">58.1</td>
<td align="center">58.2</td>
<td align="center">52.1</td>
<td align="center">54.9</td>
<td align="center">164832157</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x_legacy/164832157/model_final_d366fa.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x_legacy/164832157/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_s1x_legacy -->
<tr><td align="left"><a href="../configs/densepose_rcnn_R_101_FPN_s1x_legacy.yaml">R_101_FPN_s1x_legacy</a></td>
<td align="center">s1x</td>
<td align="center">0.390</td>
<td align="center">0.063</td>
<td align="center">4.3</td>
<td align="center">59.5</td>
<td align="center">59.3</td>
<td align="center">53.2</td>
<td align="center">56.0</td>
<td align="center">164832182</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x_legacy/164832182/model_final_10af0e.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x_legacy/164832182/metrics.json">metrics</a></td>
</tr>
</tbody></table>

```
python export.py configs/densepose_rcnn_R_50_FPN_s1x_legacy.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x_legacy/164832157/model_final_d366fa.pkl
```

```
python export.py configs/densepose_rcnn_R_101_FPN_s1x_legacy.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x_legacy/164832182/model_final_10af0e.pkl
```

### Improved Baselines, Original Fully Convolutional Head

These models use an improved training schedule and Panoptic FPN head
from [Kirillov et al, 2019](https://arxiv.org/abs/1901.02446).

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">segm<br/>AP</th>
<th valign="bottom">dp. AP<br/>GPS</th>
<th valign="bottom">dp. AP<br/>GPSm</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_rcnn_R_50_FPN_s1x -->
<tr><td align="left"><a href="../configs/densepose_rcnn_R_50_FPN_s1x.yaml">R_50_FPN_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.359</td>
<td align="center">0.066</td>
<td align="center">4.5</td>
<td align="center">61.2</td>
<td align="center">67.2</td>
<td align="center">63.7</td>
<td align="center">65.3</td>
<td align="center">165712039</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_s1x -->
<tr><td align="left"><a href="../configs/densepose_rcnn_R_101_FPN_s1x.yaml">R_101_FPN_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.428</td>
<td align="center">0.079</td>
<td align="center">5.8</td>
<td align="center">62.3</td>
<td align="center">67.8</td>
<td align="center">64.5</td>
<td align="center">66.2</td>
<td align="center">165712084</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/metrics.json">metrics</a></td>
</tr>
</tbody></table>

```
python export.py configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl
```

```
python export.py configs/densepose_rcnn_R_101_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x/165712084/model_final_c6ab63.pkl
```

### <a name="ModelZooDeepLabV3"> Improved Baselines, DeepLabV3 Head

These models use an improved training schedule, Panoptic FPN head
from [Kirillov et al, 2019](https://arxiv.org/abs/1901.02446) and DeepLabV3 head
from [Chen et al, 2017](https://arxiv.org/abs/1706.05587).

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">lr<br/>sched</th>
<th valign="bottom">train<br/>time<br/>(s/iter)</th>
<th valign="bottom">inference<br/>time<br/>(s/im)</th>
<th valign="bottom">train<br/>mem<br/>(GB)</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">segm<br/>AP</th>
<th valign="bottom">dp. AP<br/>GPS</th>
<th valign="bottom">dp. AP<br/>GPSm</th>
<th valign="bottom">model id</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: densepose_rcnn_R_50_FPN_DL_s1x -->
<tr><td align="left"><a href="../configs/densepose_rcnn_R_50_FPN_DL_s1x.yaml">R_50_FPN_DL_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.392</td>
<td align="center">0.070</td>
<td align="center">6.7</td>
<td align="center">61.1</td>
<td align="center">68.3</td>
<td align="center">65.6</td>
<td align="center">66.7</td>
<td align="center">165712097</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_s1x/165712097/model_final_0ed407.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_s1x/165712097/metrics.json">metrics</a></td>
</tr>
<!-- ROW: densepose_rcnn_R_101_FPN_DL_s1x -->
<tr><td align="left"><a href="../configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml">R_101_FPN_DL_s1x</a></td>
<td align="center">s1x</td>
<td align="center">0.478</td>
<td align="center">0.083</td>
<td align="center">7.0</td>
<td align="center">62.3</td>
<td align="center">68.7</td>
<td align="center">66.3</td>
<td align="center">67.6</td>
<td align="center">165712116</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl">model</a>&nbsp;|&nbsp;<a href="https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/metrics.json">metrics</a></td>
</tr>
</tbody></table>

```
python export.py configs/densepose_rcnn_R_50_FPN_DL_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_s1x/165712097/model_final_0ed407.pkl
```

```
python export.py configs/densepose_rcnn_R_101_FPN_DL_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_DL_s1x/165712116/model_final_844d15.pkl
```

```
@InProceedings{Guler2018DensePose,
  title={DensePose: Dense Human Pose Estimation In The Wild},
  author={R\{i}za Alp G\"uler, Natalia Neverova, Iasonas Kokkinos},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
}
```