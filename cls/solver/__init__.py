# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
https://github.com/facebookresearch/detectron2/blob/master/detectron2/solver/__init__.py
"""

from .lr_scheduler import WarmupMultiStepLR

__all__ = [k for k in globals().keys() if not k.startswith("_")]
