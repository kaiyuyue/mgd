#!/usr/bin/env python

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import norm
from ortools.linear_solver import pywraplp
from ortools.graph import pywrapgraph


__all__ = [
    'MGDistiller',
    'get_margin_from_BN',
    'distillation_loss'
]


class MGDistiller(nn.Module):
    def __init__(self, model_t, model_s, ignore_inds=[]):
        super(MGDistiller, self).__init__()

        self.model_t = model_t
        self.model_s = model_s
        self.ignore_inds = ignore_inds

        # select reducer
        self.reducer = getattr(self, 'amp') # absolute max pooling only

        # init vars
        self.channels_t = self.model_t.encoder_q.get_channel_num()
        self.channels_s = self.model_s.encoder_q.get_channel_num()

        # init margins
        self.init_margins()

        # build nets
        norm_layer = nn.BatchNorm2d
        self.BNs = nn.ModuleList([norm_layer(32, s) if norm_layer == nn.GroupNorm else norm_layer(s) for s in self.channels_s])

        # init flow matrix
        self.init_flow()


    def init_margins(self):
        print('mgd info: init margins')
        margins = [get_margin_from_BN(bn) for bn in self.model_t.encoder_q.get_bn_before_relu()]
        for i, margin in enumerate(margins):
            self.register_buffer(
                'margin%d' % (i+1),
                margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach()
            )

    def init_flow(self):
        print('mgd info: init flow')
        reminders = 0
        self.adj_matrix = []
        for s, t in zip(self.channels_s, self.channels_t):
            self.adj_matrix.append(np.zeros((s, t)))
            reminders += t%s

        # When the number of student channels can't be divisible by
        # the number of teacher channels, we shave the reminders.
        self.shave = False if reminders == 0 else True
        print('mgd info: shave matrix ? : {}'.format(self.shave))

        self.num_tracked_imgs = 0

    def extract_feature(self, x):
        feats_t, _ = self.model_t.forward_encoder_q(x, None)
        feats_s, _ = self.model_s.forward_encoder_q(x, None)
        self.track_running_stats(feats_t, feats_s)

    def track_running_stats(self, feats_t, feats_s):
        feat_num = len(feats_t)

        for i in range(feat_num):
            if i in self.ignore_inds:
                continue

            feat_t, feat_s = feats_t[i], feats_s[i]

            b, tc = feat_t.shape[0:2]
            _, sc = feat_s.shape[0:2]

            feat_t = F.normalize(feat_t.reshape(b, tc, -1), p=2, dim=2)
            feat_s = F.normalize(feat_s.reshape(b, sc, -1), p=2, dim=2)

            cost = 2 - 2 * torch.bmm(feat_s, feat_t.transpose(1, 2))
            self.adj_matrix[i] += cost.sum(dim=0).cpu().data.numpy()

        self.num_tracked_imgs += b

    def update_flow(self):
        print('mgd info: update flow')
        feat_num = len(self.adj_matrix)

        self.guided_inds = []

        for i in range(feat_num):
            _col_ind = []

            sc, tc = self.adj_matrix[i].shape

            _adj_mat = []
            if sc != tc:
                _adj_mat = np.concatenate(
                    [self.adj_matrix[i] for _ in range(tc // sc)],
                    axis=0
                )
            else:
                _adj_mat = self.adj_matrix[i]

            cost = _adj_mat / self.num_tracked_imgs
            start = time.time()
            assignment = pywrapgraph.LinearSumAssignment()

            rows, cols = cost.shape

            # shave
            cols = rows if self.shave else cols

            for r in range(rows):
                for c in range(cols):
                    assignment.AddArcWithCost(r, c, int(1e5 * cost[r][c]))

            solve_status = assignment.Solve()
            if solve_status == assignment.OPTIMAL:
                _col_ind = [
                    assignment.RightMate(n)
                    for n in range(0, assignment.NumNodes())
                ]
                cost_sum = sum(assignment.AssignmentCost(n)
                    for n in range(0, assignment.NumNodes())
                )
            print('mgd info: solve assignment for stage {}\tflow matrix shape: {}\ttime: {:.5f}\tcost: {:.5f}'.format(
                i, cost.shape, time.time()-start, 1e-5 * cost_sum)
            )

            flow_inds = torch.from_numpy(np.asarray(_col_ind)).long().cuda()

            # broadcast to all gpus
            torch.distributed.broadcast(flow_inds, src=0)

            self.guided_inds.append(flow_inds)

    def amp(self, i, feats_t, feats_s, margins):
        """
        Absolute Max Pooling for channels reduction.
        """
        b, sc, h, w = feats_s[i].shape
        _, tc, _, _ = feats_t[i].shape

        groups = tc // sc

        t = []
        m = []
        for c in range(0, tc, sc):
            if c == (tc // sc) * sc and self.shave:
                continue

            t.append(feats_t[i][:, self.guided_inds[i][c:c+sc].detach(), :, :])
            m.append(margins[:, self.guided_inds[i][c:c+sc].detach(), :, :])

        t = torch.stack(t, dim=2)
        m = torch.stack(m, dim=2)

        t = t.reshape(b, sc, groups, -1)
        m = m.reshape(1, sc, groups, -1)

        t_inds = torch.argmax(torch.abs(t), dim=2)

        t = t.gather(2, t_inds.unsqueeze(2))
        m = m.mean(dim=2)

        t = t.reshape(b, sc, h, w)
        m = m.reshape(1, sc, 1, 1)

        return t, m


    def forward(self, im_q, im_k):
        """
        Input:
            img_q: a batch of query images
            img_k: a batch of key images
        Output:
            logits, targets
        """

        feats_t, _ = self.model_t.forward_encoder_q(im_q, im_k)
        logits_s, target_s, feats_s = self.model_s(im_q, im_k)
        feat_num = len(feats_t)

        loss_factors = [2 ** (feat_num - i - 1) for i in range(feat_num)]
        loss_distill = 0
        for i in range(feat_num):
            if i in self.ignore_inds:
                continue

            # margins
            margins = getattr(self, 'margin%d' % (i+1))

            # bn for student features
            feats_s[i] = self.BNs[i](feats_s[i])

            # reduce channels
            t, m = self.reducer(i, feats_t, feats_s, margins)

            # accumulate loss
            loss_distill += distillation_loss(feats_s[i], t.detach(), m) / loss_factors[i]

        return logits_s, target_s, loss_distill


def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()


def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

