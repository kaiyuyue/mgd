#!/usr/bin/env python

import os
import pickle as pkl
import sys
import torch

if __name__ == "__main__":
    t_input = sys.argv[1]
    s_input = sys.argv[2]

    newmodel = {}
    for input, body in zip([t_input, s_input], ['t_net', 's_net']):
        ftype = os.path.basename(input).split('.')[-1]
        if ftype == 'pkl':
            with open(input, 'rb') as f:
                obj = pkl.load(f)
        elif ftype == 'pth':
            obj = torch.load(input, map_location="cpu")
        else:
            raise('Unknown file type!')

        if 'model' in obj.keys():
            obj = obj["model"]

        for k, v in obj.items():
            old_k = k
            if body == 's_net':
                k = '{}.backbone.bottom_up.{}'.format(body, old_k)
            else:
                k = '{}.{}'.format(body, old_k)
            print('{0:<55} -> {1:<55}'.format(old_k, k))

            if isinstance(v, torch.Tensor):
                v = v.cpu()

            newmodel[k] = v

    res = {"model": newmodel, "__author__": "TORCHVISION", "matching_heuristics": True}

    with open(sys.argv[3], "wb") as f:
        pkl.dump(res, f)
