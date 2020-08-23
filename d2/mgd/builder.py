#!/usr/bin/env python

import torch
import torch.nn as nn

from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch import retinanet
from detectron2.layers.batch_norm import NaiveSyncBatchNorm

from .config import mgd_teacher_config, mgd_student_config
from .mgd import MGDistiller, SMDistiller


def freeze_partial_modules(module):
    # disable grads for teacher
    for n, m in module.named_modules():
        if not 't_net' in n:
            continue
        for n, p in m.named_parameters():
            p.requires_grad = False

    return module


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    # build teacher
    mgd_teacher_config(cfg)
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    t_net = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    t_net.to(torch.device(cfg.MODEL.DEVICE))

    # build student
    mgd_student_config(cfg)
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    s_net = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    s_net.to(torch.device(cfg.MODEL.DEVICE))

    # freeze cfg
    cfg.freeze()

    # set reducer
    norm_reducers = ['amp', 'rd', 'sp']
    spec_reducers = ['sm']
    assert cfg.MGD.REDUCER in norm_reducers + spec_reducers

    # build distiller
    distiller = MGDistiller if cfg.MGD.REDUCER in norm_reducers \
           else SMDistiller

    model = distiller(
        t_net,
        s_net,
        ignore_inds=cfg.MGD.IGNORE_INDS,
        reducer=cfg.MGD.REDUCER,
        sync_bn=cfg.MGD.SYNC_BN,
        preReLU=cfg.MGD.PRERELU,
        distributed=True,
        det=True
    )
    model = freeze_partial_modules(model)
    model.to(torch.device(cfg.MODEL.DEVICE))

    return model

