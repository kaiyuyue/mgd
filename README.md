Matching Guided Distillation
===

> [**Project Webpage**](http://kaiyuyue.com/mgd) | [**Paper**](https://arxiv.org/abs/2008.09958)

This implementation is based on [the official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet), 
which supports two training modes [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel) (DP) and [DistributedDataParallel](https://pytorch.org/docs/stable/distributed.html) (DDP).
MGD for object detection is also re-implemented in [Detectron2](https://github.com/facebookresearch/detectron2) as an external project.

![introfig](.github/intro@mgd.light.svg)

Note: **T** : teacher feature tensors. **S** : student feature tensors. *dp* : distance function for distillation. *Ci*: i-th channel.

## Requirements

- Linux with Python ≥ 3.6
- PyTorch ≥ 1.4.0 
- Google Optimization Tools ([OR-Tools](https://developers.google.com/optimization)). Install it by ```pip install ortools```.

## Preparation

Prepare ImageNet-1K dataset following [the official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).
[CUB-200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) should have the same data structure.

<details>
  <summary>Directory Structure</summary>

```bash
`-- path/to/${ImageNet-1K}/root/folder
    `-- train
    |   |-- n01440764
    |   |-- n01734418
    |   |-- ...
    |   |-- n15075141
    `-- valid
    |   |-- n01440764
    |   |-- n01734418
    |   |-- ...
    |   |-- n15075141
`-- path/to/${CUB-200}/root/folder
    `-- train
    |   |-- 001.Black_footed_Albatross
    |   |-- 002.Laysan_Albatross
    |   |-- ...
    |   |-- 200.Common_Yellowthroat
    `-- valid
    |   |-- 001.Black_footed_Albatross
    |   |-- 002.Laysan_Albatross
    |   |-- ...
    |   |-- 200.Common_Yellowthroat
```
</details>

## Training

We take the distillation with MGD on ImageNet-1K as our example here to illustrate how to train a base model and how to distill a student using MGD.

- **GPU Environment** 

To control how many and which gpus to use for training or evaluation, set

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

- **Base Training** 

To train the base models, for example MobileNet-V1, run the script `main_base.py`

```bash
python main_base.py \
    [your imagenet-1k with train and valid folders] \
    --arch mobilenet_v1 
```

- **MGD Training** 

MGD settings are same as settings for base training.
To do distilling MobileNet-V1 by ResNet50 with MGD, run the script `main_mgd.py`

```bash
python main_mgd.py \
    [your imagenet-1k with train and valid folders] \
    --arch mobilenet_v1 \
    --distiller mgd \
    --mgd-reducer amp \
    --mgd-update-freq 1
```

- **with KD**

Since MGD is lightweight and parameter-free, it can be used together with other methods, such as the classic [KD](https://arxiv.org/abs/1503.02531).
Thus MGD distills student using intermediate feature maps, KD distills student using final output logits.
If one would like to enable MGD to be working with KD for training, additionally set

```bash
    --mgd-with-kd 1
```

- **DDP Mode** 

The default training mode is DP as same as the mode in paper. 
The DDP training mode is also supported in this code but only for experimental and research purpose. 
To run within DDP mode, additionally set

```bash
    --world-size 1 \
    --rank 0 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed
```

In DDP mode, there are some differences in the values of flow matrices on each rank because the observed statistics of `torch.nn.BatchNorm` are different.
This doesn't affect the MGD training.
If one would like to force the batched statistics and the affine parameters of norm layers to be the same for all the ranks, please set

```bash
    --sync-bn 1
```

## Evaluation

This code supports evaluation for both teacher and student model at the same time. Enable `--evaluate` and set

```bash
python main_base.py \
    [your imagenet-1k with train and valid folders] \
    --arch mobilenet_v1 \
    --teacher-resume [your teacher checkpoint] \
    --student-resume [your student checkpoint] \
    --evaluate
```

## Transfer Learning

| **model** | **method** | **best top1 err.** | **top5 err.** |
|:---|:---|:---:|:---:|
| ResNet-50    | Teacher | 20.02 | 6.06 |
| MobileNet-V2 | Student | 24.61 | 7.56 |
| | MGD - AMP | 20.47 | 5.23 |
| ShuffleNet-V2| Student | 31.39 | 10.9 |
| | MGD - AMP | 25.95 | 7.46 |

<details>
  <summary>Training Script on CUB-200</summary>

```bash
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python main_mgd.py \
    [path/to/${CUB-200}/root/folder] \
    --arch mobilenet_v2 \ # or shufflenet_v2
    --epochs 120 \
    --batch-size 64 \
    --learning-rate 0.01 \
    --distiller mgd \
    --mgd-reducer amp \
    --mgd-update-freq 2 \
    --use-pretrained 1 \
    --teacher-resume [path/to/cub/teacher/pth]
```
</details>

MobileNet-V2 has a same performance with teacher on CUB-200, but ShuffleNet-V2 doesn't.
Here we boost the performance for ShuffleNet-V2 using MGD and KD together.

| **model** | **method** | **best top1 err.** | **top5 err.** |
|:---|:---|:---:|:---:|
| ResNet-50    | Teacher | 20.02 | 6.06 |
| ShuffleNet-V2| Student | 31.39 | 10.9 |
| | MGD - AMP + KD | 25.18 | 7.870 |

## Large-Scale Classification

| **model** | **method** | **best top1 err.** | **top5 err.** |
|:---|:---|:---:|:---:|
| ResNet-50    | Teacher | 23.85 | 7.13 |
| MobileNet-V1 | Student | 31.13 | 11.24 |
| | MGD - AMP | 28.53 | 9.67 |

<details>
  <summary>Training Script on ImageNet-1K</summary>

```bash
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main_mgd.py \
    [path/to/${ImageNet-1K}/root/folder] \
    --arch mobilenet_v1 \
    --epochs 120 \
    --print-freq 10 \
    --batch-size 256 \
    --learning-rate 0.1 \
    --distiller mgd \
    --mgd-reducer amp \
    --mgd-update-freq 1 \
    --warmup 1
```
</details>

## Object Detection

See [./d2](./d2).

## Acknowledgement

We learn and use some part of codes from following projects. We thank these excellent works:

* [A Comprehensive Overhaul of Feature Distillation](https://github.com/clovaai/overhaul-distillation), ICCV'19.
* [Detectron2](https://github.com/facebookresearch/detectron2). FAIR's next-generation platform for object detection and segmentation.

## License

MIT. See [LICENSE](./LICENSE) for details.
