#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Modified from https://github.com/facebookresearch/moco/blob/master/detection/convert-pretrain-to-detectron2.py
"""

import pickle as pkl
import sys
import torch

if __name__ == "__main__":
    net_names = ['s_net', 't_net']
    if len(sys.argv) == 4 and sys.argv[3] == '--eval-teacher':
        net_names[0], net_names[1] = net_names[1], net_names[0]

    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")
    if 'model' in obj.keys():
        obj = obj["model"]

    newmodel = {}
    for k, v in obj.items():
        old_k = k

        if net_names[1] in k:
            continue

        k = old_k.replace('{}.'.format(net_names[0]), '')

        # print('{0:<55} -> {1:<55}'.format(old_k, k))
        newmodel[k] = v.detach().numpy()

    res = {"model": newmodel, "__author__": "TORCHVISION", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
