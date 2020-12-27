# 中文检测DBNet

This repo provides training and demo code of DBnet  text detection on Chinese characters.



## Data Preparation

Training data: prepare a text `train.txt` in the following format, use '\t' as a separator
```
./datasets/train/img/001.jpg	./datasets/train/gt/001.txt
```

Validation data: prepare a text `test.txt` in the following format, use '\t' as a separator
```
./datasets/test/img/001.jpg	./datasets/test/gt/001.txt
```
- Store images in the `img` folder
- Store groundtruth in the `gt` folder

The groundtruth can be `.txt` files, with the following format:
```
x1, y1, x2, y2, x3, y3, x4, y4, annotation
```


## Train
Simply run:

```
./single_gpu_train.sh
```

This will train on ICPR dataset.



## Demo

We have a quick demo to visualize detection result:

```
 python3 demo.py --model_path output/DBNet_resnet18_FPN_DBHead/checkpoint/model_best.pth --data ./imgs/
```

The pretrained model can be download from manaai.cn.



<h2 id="Performance">Performance</h2>

### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4)
only train on ICDAR2015 dataset

| Method                   | image size (short size) |learning rate | Precision (%) | Recall (%) | F-measure (%) | FPS |
|:--------------------------:|:-------:|:--------:|:--------:|:------------:|:---------------:|:-----:|
| SynthText-Defrom-ResNet-18(paper)  | 736 |0.007 | 86.8 | 78.4 | 82.3 | 48 |
| ImageNet-resnet18-FPN-DBHead  |736 |1e-3| 87.03 | 75.06 | 80.6 | 43 |
| ImageNet-Defrom-Resnet18-FPN-DBHead  |736 |1e-3| 88.61 | 73.84 | 80.56 | 36 |
| ImageNet-resnet50-FPN-DBHead  |736 |1e-3| 88.06 | 77.14 | 82.24 | 27 |
| ImageNet-resnest50-FPN-DBHead  |736 |1e-3| 88.18 | 76.27 | 81.78 | 27 |

### 

## Reference

1. https://arxiv.org/pdf/1911.08947.pdf
2. https://github.com/WenmuZhou/PANet.pytorch
3. https://github.com/MhLiao/DB



## Copyright

all rights belongs to WenmuZhou && MANAAI authors.