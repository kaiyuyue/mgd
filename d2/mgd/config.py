#!/usr/bin/env python
from detectron2.config import CfgNode as CN

def mgd_teacher_config(cfg):
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.ENABLE_STAGE_LAST_NORM = False

def mgd_student_config(cfg):
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.BACKBONE.FREEZE_AT = 1
    cfg.MODEL.RESNETS.DEPTH = 18
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    cfg.MODEL.RESNETS.ENABLE_STAGE_LAST_NORM = True

def add_mgd_config(cfg):
    cfg.SOLVER.NESTEROV = True
    cfg.INPUT.FORMAT = "RGB"

    cfg.MGD = CN()
    cfg.MGD.REDUCER = 'amp'
    # [c2, c3, c4, c5, p3, p4, p5, p6, p7]
    cfg.MGD.IGNORE_INDS = [4, 5, 6, 7, 8]
    cfg.MGD.SYNC_BN = False
    cfg.MGD.PRERELU = True
    cfg.MGD.UPDATE_FREQ = [5000, 31000] # After 20000 iters of 1x schedule, we slow down the update frequency.
    cfg.MGD.UPDATE_SLOW_STEP = [20000]
    cfg.MGD.UPDATE_ITER = 1000
