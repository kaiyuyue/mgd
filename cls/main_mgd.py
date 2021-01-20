#!/usr/bin/env python

import argparse
import os
import random
import shutil
import time
import warnings
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import mgd.builder
import mgd.sampler
import solver

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch MGD Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='student model architecture')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# args for lr scheduler
parser.add_argument('--lr-drop-ratio', default=0.1, type=float,
                    help='the learning rate drop ratio')
parser.add_argument('--lr-drop-epochs', default=[31, 61, 81], type=int,
                    nargs='+',
                    help='the learning rate drop epoch list (default: 31 61 81)')

# args for distillation
parser.add_argument('--use-pretrained', default=0, type=int,
                    help='use IN-1K pretrained weight for student model')
parser.add_argument('--loss-factor', default=1e4, type=float,
                    help='the factor for distillation loss (default: 1e4 for input size 224)')

# args for MGD
parser.add_argument('--distiller', default='mgd', type=str,
                    help='distiller for building the model (default: mgd)')
parser.add_argument('--mgd-reducer', default='amp', type=str,
                    help='mgd reducer for channels reduction: '
                         'sm : sparse matching | '
                         'rd : random drop | '
                         'amp : absolute max pooling | '
                         'sp : simple pooling '
                         '(default: amp)')
parser.add_argument('--mgd-update-freq', default=2, type=int,
                    help='update frequency for flowe matrix (default: 2)')
parser.add_argument('--mgd-with-kd', default=0, type=int,
                    help='use mgd and kd together (default: 0)')

# args for resume
parser.add_argument('--student-resume', default=None, type=str,
                    help='load checkpoint for student model')
parser.add_argument('--teacher-resume', default=None, type=str,
                    help='load checkpoint for teacher model')

# args for bn
parser.add_argument('--sync-bn', default=0, type=int,
                    help='convert student bn into sync bn in DDP mode (default: 0)')

# args for warmup
parser.add_argument('--warmup', default=0, type=int,
                    help='warmup usage (default: 0)')
parser.add_argument('--warmup-epochs', default=10, type=int,
                    help='epochs/iterations for warmup (default: 10 epochs)')

best_acc1 = 0
best_acc5 = 0

def main():
    args = parser.parse_args()

    if args.sync_bn:
        assert TORCH_VERSION > (1, 5), \
        'In PyTorch <= 1.5, `nn.SyncBatchNorm` has incorrect gradient ' \
        'when the batch size on each worker is different.' \
        'Please upgrade your PyTorch.' \
        'Or you can use NaiveSyncBatchNorm in Detectron2. ' \
        'https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/batch_norm.py#L168'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1, best_acc5
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.arch == 'mobilenet_v1':
        from models import resnet, mobilenetv1
        t_net = resnet.resnet50(pretrained=True)
        s_net = mobilenetv1.mobilenet_v1()
        ignore_inds = []
    elif args.arch == 'mobilenet_v2':
        from models import resnet, mobilenetv2
        t_net = resnet.resnet50(pretrained=True)
        s_net = mobilenetv2.mobilenet_v2(
            pretrained=bool(args.use_pretrained)
        )
        ignore_inds = [0]
    elif args.arch == 'resnet50':
        from models import resnet
        t_net = resnet.resnet152(pretrained=True)
        s_net = resnet.resnet50(
            pretrained=bool(args.use_pretrained)
        )
        ignore_inds = []
    elif args.arch == 'shufflenet_v2':
        from models import resnet, shufflenetv2
        t_net = resnet.resnet50(pretrained=True)
        s_net = shufflenetv2.shufflenet_v2_x1_0(
            pretrained=bool(args.use_pretrained)
        )
        ignore_inds = [0]
    else:
        raise ValueError

    if args.distiller == 'mgd':
        # normal and special reducers
        norm_reducers = ['amp', 'rd', 'sp']
        spec_reducers = ['sm']
        assert args.mgd_reducer in norm_reducers + spec_reducers

        # create distiller
        distiller = mgd.builder.MGDistiller if args.mgd_reducer in norm_reducers \
               else mgd.builder.SMDistiller

        d_net = distiller(
            t_net,
            s_net,
            ignore_inds=ignore_inds,
            reducer=args.mgd_reducer,
            sync_bn=args.sync_bn,
            with_kd=args.mgd_with_kd,
            preReLU=True,
            distributed=args.distributed
        )
    else:
        raise NotImplementedError

    # model size
    print('the number of teacher model parameters: {}'.format(
        sum([p.data.nelement() for p in t_net.parameters()]))
    )
    print('the number of student model parameters: {}'.format(
        sum([p.data.nelement() for p in s_net.parameters()]))
    )
    print('the total number of model parameters: {}'.format(
        sum([p.data.nelement() for p in d_net.parameters()]))
    )

    # dp convert
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            t_net.cuda(args.gpu)
            s_net.cuda(args.gpu)
            d_net.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            if args.sync_bn:
                s_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(s_net)
            t_net = torch.nn.parallel.DistributedDataParallel(t_net, find_unused_parameters=True, device_ids=[args.gpu])
            s_net = torch.nn.parallel.DistributedDataParallel(s_net, find_unused_parameters=True, device_ids=[args.gpu])
            d_net = torch.nn.parallel.DistributedDataParallel(d_net, find_unused_parameters=True, device_ids=[args.gpu])
        else:
            t_net.cuda()
            s_net.cuda()
            d_net.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            if args.sync_bn:
                s_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(s_net)
            t_net = torch.nn.parallel.DistributedDataParallel(t_net, find_unused_parameters=True)
            s_net = torch.nn.parallel.DistributedDataParallel(s_net, find_unused_parameters=True)
            d_net = torch.nn.parallel.DistributedDataParallel(d_net, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        t_net = t_net.cuda(args.gpu)
        s_net = s_net.cuda(args.gpu)
        d_net = d_net.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        t_net = torch.nn.DataParallel(t_net).cuda()
        s_net = torch.nn.DataParallel(s_net).cuda()
        d_net = torch.nn.DataParallel(d_net).cuda()

    # define loss function (criterion), optimizer and lr_scheduler
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    model_params = list(s_net.parameters()) + list(d_net.module.BNs.parameters())
    optimizer = torch.optim.SGD(model_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    # warmup setting
    if args.warmup:
        args.epochs += args.warmup_epochs
        args.lr_drop_epochs = list(
            np.array(args.lr_drop_epochs) + args.warmup_epochs
        )
    lr_scheduler = build_lr_scheduler(optimizer, args)

    # optionally resume from a checkpoint
    load_checkpoint(t_net, args.teacher_resume, args)
    load_checkpoint(s_net, args.student_resume, args)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'valid')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(validdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.distributed:
        extra_sampler = mgd.sampler.ExtraDistributedSampler(train_dataset)
    else:
        extra_sampler = None

    extra_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=(extra_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=extra_sampler)

    print('=> evaluate teacher model')
    validate(valid_loader, t_net, criterion, args)
    print('=> evaluate student model')
    validate(valid_loader, s_net, criterion, args)
    if args.evaluate:
        return

    if args.distiller == 'mgd':
        mgd_update(extra_loader, d_net, args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, d_net, criterion, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1, acc5 = validate(valid_loader, s_net, criterion, args)

        # update flow matrix for the next round
        if args.distiller == 'mgd' and (epoch+1)%args.mgd_update_freq == 0:
            mgd_update(extra_loader, d_net, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = acc5 if is_best else best_acc5

        print(' * - Best - Err@1 {acc1:.3f} Err@5 {acc5:.3f}'
              .format(acc1=(100-best_acc1), acc5=(100-best_acc5)))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            filename = '{}.pth'.format(args.arch)
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': s_net.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': acc5,
                'optimizer': optimizer.state_dict(),
                },
                is_best,
                filename
            )
        lr_scheduler.step()
        gc.collect()


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}] LR: {:.6f}\t".format(epoch, lr_scheduler.get_last_lr()[0]))

    # switch mode
    model.train()
    model.module.t_net.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output and loss
        output, d_loss = model(images)
        loss = criterion(output, target)

        # add kd loss
        if args.mgd_with_kd:
            d_loss, k_loss = d_loss
            loss += k_loss.mean()

        # add distillation loss
        loss += d_loss.sum() / args.batch_size / args.loss_factor

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(valid_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(valid_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(valid_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def mgd_update(extra_loader, model, args):
    # switch to evaluate mode
    model.eval()

    # init flow
    model.module.init_flow()

    with torch.no_grad():
        for i, (images, _) in enumerate(extra_loader):
            images = images.cuda(args.gpu, non_blocking=True)

            # running for tracking status
            model.module.extract_feature(images)

            # break for ImageNet-1K
            if args.batch_size * i > 20000:
                break

        # update transpose/flow matrix
        model.module.update_flow()


def save_checkpoint(args, state, is_best, filename):
    """Save checkpoint"""
    root = 'ckpt'
    task = '.'.join([
        os.path.basename(args.data),
        'distillation',
        args.arch
    ])
    task = '.'.join([task, args.distiller])
    if args.distiller == 'mgd':
        task = '.'.join([task, args.mgd_reducer])
    root = os.path.join(root, task)

    os.makedirs(root, exist_ok=True)
    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(root, filename),
            os.path.join(root, 'best.{}'.format(filename))
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def load_checkpoint(model, ckpt_path, args):
    """Load checkpoint"""
    if ckpt_path:
        if os.path.isfile(ckpt_path):
            print("=> loading checkpoint '{}'".format(ckpt_path))
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(ckpt_path, checkpoint['epoch']))
            # release occupation
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(ckpt_path))
            raise

def build_lr_scheduler(optimizer, args):
    """Build the LR scheduler for training.
    """
    if args.warmup:
        lr_scheduler =solver.lr_scheduler.WarmupMultiStepLR(
            optimizer,
            milestones=args.lr_drop_epochs,
            gamma=args.lr_drop_ratio,
            warmup_factor=0.1,
            warmup_iters=args.warmup_epochs,
            last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.lr_drop_epochs,
            gamma=args.lr_drop_ratio,
            last_epoch=-1)
    return lr_scheduler


if __name__ == '__main__':
    main()

