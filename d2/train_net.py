#!/usr/bin/env python

import logging
import os
import torch
import time

from torch.nn.parallel import DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch
)
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils import comm
from detectron2.data import (
    MetadataCatalog,
    get_detection_dataset_dicts,
    build_batch_data_loader
)
from detectron2.data import transforms as T
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import TrainingSampler
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)

from mgd import add_mgd_config
from mgd import build_model

__LOGGER_NAME__ = 'detectron2'

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MGD.
    """
    def mgd_update_flow(self):
        # switch mode
        self.model.eval()
        # init flow
        self.model.module.init_flow()
        # iter count
        iters = 0
        # extract features
        logger = logging.getLogger(__LOGGER_NAME__)
        with torch.no_grad():
            for data in self._data_loader_iter:
                # running for tracking status
                self.model.module.extract_feature(data)
                # break for COCO
                if iters >= self.cfg.MGD.UPDATE_ITER:
                    break
                # count
                iters += 1
                if iters % 20 == 0:
                    logger.info('mgd: track_running_stats\t iter: {}'.format(str(iters).zfill(5)))
        # update transpose/flow matrix
        self.model.module.update_flow()

    def mgd_update_step(self):
        assert len(self.cfg.MGD.UPDATE_SLOW_STEP) == 1
        assert len(self.cfg.MGD.UPDATE_FREQ) - 1 == \
               len(self.cfg.MGD.UPDATE_SLOW_STEP)

        # for the starting of training or resuming
        if self.iter == 0 or (self.iter != 0 and self.resume):
            self.model.module.init_margins()
            self.mgd_update_flow()
            self.resume = None
        # interval update and reset the step counter
        else:
            update_freq = self.cfg.MGD.UPDATE_FREQ[0] \
                if self.iter < self.cfg.MGD.UPDATE_SLOW_STEP[0] \
                else self.cfg.MGD.UPDATE_FREQ[1]
            if self.iter % update_freq == 0 and self.iter != 0:
                self.mgd_update_flow()

    def switch_into_train_mode(self):
        if not self.model.training:
            # switch into right modes
            self.model.train()
            self.model.module.t_net.eval()
            self.model.module.s_net.train()

    def run_step(self):
        # run step for mgd update
        self.mgd_update_step()

        # switch into train mode
        self.switch_into_train_mode()

        # assert
        assert self.model.training, "[MGDTrainer] model was changed to eval mode!"

        # data
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # loss
        loss_dict = self.model(data)[0]
        loss_dict['loss_distill'] = loss_dict['loss_distill'] / self.cfg.SOLVER.IMS_PER_BATCH
        losses = sum(loss_dict.values())

        # backward
        self.optimizer.zero_grad()
        losses.backward()

        # use a new stream so the ops don't wait for DDP
        with torch.cuda.stream(
            torch.cuda.Stream()
        ) if losses.device.type == "cuda" else _nullcontext():
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)
        self.optimizer.step()

    def wrap_model_with_ddp(self, cfg, model):
        # work with PR: https://github.com/facebookresearch/detectron2/pull/1820
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=False,
                find_unused_parameters=True
            )
        return model

    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        self.resume = resume

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__LOGGER_NAME__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["coco"]:
            evaluator_list.append(
                COCOEvaluator(dataset_name, cfg, True, output_folder)
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    add_mgd_config(cfg)
    # NOTE: freeze cfg after setting done with teacher, student, and distiller
    # cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg).s_net
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
