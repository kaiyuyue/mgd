MGD in Unsupervised Learning
===

This experiment is an extension of the original paper.
MGD can naturally work with current unsupervised learning frameworks, e.g., [Momentum Contrast](https://github.com/facebookresearch/moco) (MoCo) and [Simple Siamese Learning](https://github.com/facebookresearch/simsiam) (SimSiam).
In this repo, we initially investigate MoCo-v2 training with MGD and work in progress on other parts.

## Environments

- PyTorch 1.8.1 

## Data Preparation

Prepare ImageNet-1K dataset following [the official PyTorch ImageNet training code](https://github.com/pytorch/examples/tree/master/imagenet).

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
```
</details>

## Code Preparation

`cp -r ../mgd/sampler.py mgd`

## Unsupervised Training with MGD

Please download the pre-trained weight (md5: `59fd9945`, epochs: `200`) of ResNet-50 from [MoCo-v2 Models](https://github.com/facebookresearch/moco#models) and then load it with the arg of `--resume`.

To do unsupervised pre-training of a ResNet-18 model with MGD on ImageNet in an 8-gpu machine, run:

```bash
python main_moco_mgd.py \
  -a resnet18 \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --resume moco_v2_200ep_pretrain.pth.tar \
  [your imagenet-folder with train and val folders]
```

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">method</th>
<th valign="bottom">model</th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">training<br/>logs</th>
<!-- TABLE BODY -->
<tr><td align="left">MGD</td>
<td align="left">ResNet-50 distills ResNet-34</td>
<td align="center">200</td>
<td align="center"><a href="https://pan.baidu.com/s/1zyg6HCfIR_buiYaFeK7QhA">Baidu Pan</a> [ bkr5 ]</td>
</tr>
<tr><td align="left">MGD</td>
<td align="left">ResNet-50 distills ResNet-18</td>
<td align="center">200</td>
<td align="center"><a href="https://pan.baidu.com/s/1UTshhObAEbRy6h1gY4ixhQ">Baidu Pan</a> [ jbcv ]</td>
</tr>
</tbody></table>

**Note**: 

- The MGD distiller is engined by the AMP -- absolute max pooling.
- The teacher is ResNet-50 [in deafult](https://github.com/KaiyuYue/mgd/blob/master/unsup/main_moco_mgd.py#L178).
- The hyper-parameters of MGD, such as loss factors, are the same as supervised training. We did not search hyper-parameters. But according to training logs, we believe performances can be better with tunning hyper-parameters, for example, increasing the [factor](https://github.com/KaiyuYue/mgd/blob/master/unsup/main_moco_mgd.py#L363) from `1e4` to `1e2`.


## Linear Classification

Same as [linear classification](https://github.com/facebookresearch/moco#linear-classification) of MoCo-v2.
Linear classification results on ImageNet using this repo with 8 NVIDIA TITAN Xp GPUs:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">method</th>
<th valign="bottom">model</th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">MoCo v2<br/>top-1 acc.</th>
<th valign="bottom">MoCo v2<br/>top-5 acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">Teacher</td>
<td align="left">ResNet-50</td>
<td align="center">200</td>
<td align="center">67.5</td>
<td align="center">-</td>
</tr>
<tr><td align="left">Student</td>
<td align="left">ResNet-34</td>
<td align="center">200</td>
<td align="center">57.2</td>
<td align="center">81.5</td>
</tr>
<tr><td align="left">MGD</td>
<td align="left">ResNet-34</td>
<td align="center">200</td>
<td align="center">58.5</td>
<td align="center">82.7</td>
</tr>
<tr><td align="left">Student</td>
<td align="left">ResNet-18</td>
<td align="center">200</td>
<td align="center">52.5</td>
<td align="center">77.0</td>
</tr>
<tr><td align="left">MGD</td>
<td align="left">ResNet-18</td>
<td align="center">200</td>
<td align="center">53.6</td>
<td align="center">78.7</td>
</tr>
</tbody></table>

## Update Schedule

The schedule for updating MGD matching matrix is different with that in the original paper.
We scale it with a log function, i.e., we update matching matrix at the epoch of `[1, 2, 3, 6, 9, 15, 26, 43, 74, 126]`.
