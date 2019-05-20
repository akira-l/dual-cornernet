import numpy as np
import torch
import torch.nn as nn

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr

from .kp_utils import _tranpose_and_gather_feat, _decode
from .kp_utils import _sigmoid, _ae_loss, _regr_loss, _neg_loss
from .kp_utils import make_tl_layer, make_br_layer, make_kp_layer
from .kp_utils import make_pool_layer, make_unpool_layer
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer

import pdb

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class kp(nn.Module):
    def __init__(
        self, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(kp, self).__init__()

        self.nstack    = nstack
        self._decode   = _decode

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])


        # begin to change layers here

        self.sub_tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.sub_br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        self.obj_tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.obj_br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.sub_tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.sub_br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])


        self.obj_rela_tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.obj_rela_br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])


        ## tags
        self.sub_tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.sub_br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])

        self.obj_tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])
        self.obj_br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])

        for sub_tl_heat, sub_br_heat, obj_tl_heat, obj_br_heat in zip(self.sub_tl_heats, self.sub_br_heats, self.obj_tl_heats, self.obj_br_heats):
            sub_tl_heat[-1].bias.data.fill_(-2.19)
            sub_br_heat[-1].bias.data.fill_(-2.19)
            obj_tl_heat[-1].bias.data.fill_(-2.19)
            obj_br_heat[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        
        

        self.sub_tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.sub_br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.obj_tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.obj_br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])

        self.relu = nn.ReLU(inplace=True)
    

    def _train(self, *xs):
        image   = xs[0]
        # subject point
        sub_tl_inds = xs[1]
        sub_br_inds = xs[2]
        # object point
        obj_tl_inds = xs[1]
        obj_br_inds = xs[2] # need be fixed here !!!!

        inter = self.pre(image)
        outs  = []

        layers = zip(
            self.kps, self.cnvs,

            self.sub_tl_cnvs, self.sub_br_cnvs,
            self.sub_tl_heats, self.sub_br_heats,
            self.sub_tl_tags, self.sub_br_tags,
            self.sub_tl_regrs, self.sub_br_regrs,

            self.obj_tl_cnvs, self.obj_br_cnvs, 
            self.obj_tl_heats, self.obj_br_heats, 
            self.obj_tl_tags, self.obj_br_tags, 
            self.obj_tl_regrs, self.obj_br_regrs
        )
        counter = 0
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]

            sub_tl_cnv_, sub_br_cnv_   = layer[2:4]
            sub_tl_heat_, sub_br_heat_ = layer[4:6]
            sub_tl_tag_, sub_br_tag_   = layer[6:8]
            sub_tl_regr_, sub_br_regr_ = layer[8:10]

            obj_tl_cnv_, obj_br_cnv_ = layer[10:12]
            obj_tl_heat_, obj_br_heat_ = layer[12:14]
            obj_tl_tag_, obj_br_tag_ = layer[14:16]
            obj_tl_regr_, obj_br_regr_ = layer[16:18]
  
            counter += 1

            kp  = kp_(inter)
            cnv = cnv_(kp)

            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)

            sub_tl_heat, sub_br_heat = sub_tl_heat_(sub_tl_cnv), sub_br_heat_(sub_br_cnv)

            sub_tl_tag, sub_br_tag   = sub_tl_tag_(sub_tl_cnv),  sub_br_tag_(sub_br_cnv)
            sub_tl_regr, sub_br_regr = sub_tl_regr_(sub_tl_cnv), sub_br_regr_(sub_br_cnv)

            sub_tl_tag  = _tranpose_and_gather_feat(sub_tl_tag, sub_tl_inds)
            sub_br_tag  = _tranpose_and_gather_feat(sub_br_tag, sub_br_inds)

            sub_tl_regr = _tranpose_and_gather_feat(sub_tl_regr, sub_tl_inds)
            sub_br_regr = _tranpose_and_gather_feat(sub_br_regr, sub_br_inds)


            obj_tl_heat, obj_br_heat = obj_tl_heat_(obj_tl_cnv), obj_br_heat_(obj_br_cnv)

            obj_tl_tag, obj_br_tag   = obj_tl_tag_(obj_tl_cnv),  obj_br_tag_(obj_br_cnv)
            obj_tl_regr, obj_br_regr = obj_tl_regr_(obj_tl_cnv), obj_br_regr_(obj_br_cnv)

            obj_tl_tag  = _tranpose_and_gather_feat(obj_tl_tag, obj_tl_inds)
            obj_br_tag  = _tranpose_and_gather_feat(obj_br_tag, obj_br_inds)

            obj_tl_regr = _tranpose_and_gather_feat(obj_tl_regr, obj_tl_inds)
            obj_br_regr = _tranpose_and_gather_feat(obj_br_regr, obj_br_inds)



            outs += [sub_tl_heat, sub_br_heat, sub_tl_tag, sub_br_tag, sub_tl_regr, sub_br_regr,
                     obj_tl_heat, obj_br_heat, obj_tl_tag, obj_br_tag, obj_tl_regr, obj_br_regr]

            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return outs

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)
        outs  = []

        
        layers = zip(
            self.kps, self.cnvs,

            self.sub_tl_cnvs, self.sub_br_cnvs,
            self.sub_tl_heats, self.sub_br_heats,
            self.sub_tl_tags, self.sub_br_tags,
            self.sub_tl_regrs, self.sub_br_regrs,

            self.obj_tl_cnvs, self.obj_br_cnvs, 
            self.obj_tl_heats, self.obj_br_heats, 
            self.obj_tl_tags, self.obj_br_tags, 
            self.obj_tl_regrs, self.obj_br_regrs
        )

        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]

            sub_tl_cnv_, sub_br_cnv_   = layer[2:4]
            sub_tl_heat_, sub_br_heat_ = layer[4:6]
            sub_tl_tag_, sub_br_tag_   = layer[6:8]
            sub_tl_regr_, sub_br_regr_ = layer[8:10]

            obj_tl_cnv_, obj_br_cnv_ = layer[10:12]
            obj_tl_heat_, obj_br_heat_ = layer[12:14]
            obj_tl_tag_, obj_br_tag_ = layer[14:16]
            obj_tl_regr_, obj_br_regr_ = layer[16:18]
  
            counter += 1

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack -1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)

                sub_tl_heat, sub_br_heat = sub_tl_heat_(sub_tl_cnv), sub_br_heat_(sub_br_cnv)

                sub_tl_tag, sub_br_tag   = sub_tl_tag_(sub_tl_cnv),  sub_br_tag_(sub_br_cnv)
                sub_tl_regr, sub_br_regr = sub_tl_regr_(sub_tl_cnv), sub_br_regr_(sub_br_cnv)

                sub_tl_tag  = _tranpose_and_gather_feat(sub_tl_tag, sub_tl_inds)
                sub_br_tag  = _tranpose_and_gather_feat(sub_br_tag, sub_br_inds)

                sub_tl_regr = _tranpose_and_gather_feat(sub_tl_regr, sub_tl_inds)
                sub_br_regr = _tranpose_and_gather_feat(sub_br_regr, sub_br_inds)


                obj_tl_heat, obj_br_heat = obj_tl_heat_(obj_tl_cnv), obj_br_heat_(obj_br_cnv)

                obj_tl_tag, obj_br_tag   = obj_tl_tag_(obj_tl_cnv),  obj_br_tag_(obj_br_cnv)
                obj_tl_regr, obj_br_regr = obj_tl_regr_(obj_tl_cnv), obj_br_regr_(obj_br_cnv)

                obj_tl_tag  = _tranpose_and_gather_feat(obj_tl_tag, obj_tl_inds)
                obj_br_tag  = _tranpose_and_gather_feat(obj_br_tag, obj_br_inds)
  
                obj_tl_regr = _tranpose_and_gather_feat(obj_tl_regr, obj_tl_inds)
                obj_br_regr = _tranpose_and_gather_feat(obj_br_regr, obj_br_inds)


                outs += [sub_tl_heat, sub_br_heat, sub_tl_tag, sub_br_tag, sub_tl_regr, sub_br_regr,
                     obj_tl_heat, obj_br_heat, obj_tl_tag, obj_br_tag, obj_tl_regr, obj_br_regr]


            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return self._decode(*outs[-12:], **kwargs)

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        stride = 12

        sub_tl_heats = outs[0::stride]
        sub_br_heats = outs[1::stride]
        sub_tl_tags  = outs[2::stride]
        sub_br_tags  = outs[3::stride]
        sub_tl_regrs = outs[4::stride]
        sub_br_regrs = outs[5::stride]

        obj_tl_heats = outs[6::stride]
        obj_br_heats = outs[7::stride]
        obj_tl_tags  = outs[8::stride]
        obj_br_tags  = outs[9::stride]
        obj_tl_regrs = outs[10::stride]
        obj_br_regrs = outs[11::stride]

        sub_gt_tl_heat = targets[0]
        sub_gt_br_heat = targets[1]
        sub_gt_mask    = targets[2]
        sub_gt_tl_regr = targets[3]
        sub_gt_br_regr = targets[4]

        obj_gt_tl_heat = targets[5]
        obj_gt_br_heat = targets[6]
        obj_gt_mask    = targets[7]
        obj_gt_tl_regr = targets[8]
        obj_gt_br_regr = targets[9]

        # focal loss
        focal_loss = 0

        sub_tl_heats = [_sigmoid(t) for t in sub_tl_heats]
        sub_br_heats = [_sigmoid(b) for b in sub_br_heats]

        obj_tl_heats = [_sigmoid(t) for t in obj_tl_heats]
        obj_br_heats = [_sigmoid(b) for b in obj_br_heats]

        sub_focal_loss += self.focal_loss(sub_tl_heats, sub_gt_tl_heat)
        sub_focal_loss += self.focal_loss(sub_br_heats, sub_gt_br_heat)

        obj_focal_loss += self.focal_loss(obj_tl_heats, obj_gt_tl_heat)
        obj_focal_loss += self.focal_loss(obj_br_heats, obj_gt_br_heat)

        # tag loss
        sub_pull_loss = 0
        sub_push_loss = 0

        for sub_tl_tag, sub_br_tag in zip(sub_tl_tags, sub_br_tags):
            sub_pull, sub_push = self.ae_loss(sub_tl_tag, sub_br_tag, sub_gt_mask)
            sub_pull_loss += sub_pull
            sub_push_loss += sub_push
        sub_pull_loss = self.pull_weight * sub_pull_loss
        sub_push_loss = self.push_weight * sub_push_loss

        obj_obj_pull_loss = 0
        obj_obj_push_loss = 0

        for obj_tl_tag, obj_br_tag in zip(obj_tl_tags, obj_br_tags):
            obj_pull, obj_push = self.ae_loss(obj_tl_tag, obj_br_tag, obj_gt_mask)
            obj_pull_loss += obj_pull
            obj_push_loss += obj_push
        obj_pull_loss = self.pull_weight * obj_pull_loss
        obj_push_loss = self.push_weight * obj_push_loss

        sub_regr_loss = 0
        for sub_tl_regr, sub_br_regr in zip(sub_tl_regrs, sub_br_regrs):
            sub_regr_loss += self.regr_loss(sub_tl_regr, sub_gt_tl_regr, sub_gt_mask)
            sub_regr_loss += self.regr_loss(sub_br_regr, sub_gt_br_regr, sub_gt_mask)
        sub_regr_loss = self.regr_weight * sub_regr_loss
        
        obj_regr_loss = 0
        for obj_tl_regr,obj_br_regr in zip(obj_tl_regrs, obj_br_regrs):
            obj_regr_loss += self.regr_loss(obj_tl_regr, obj_gt_tl_regr, obj_gt_mask)
            obj_regr_loss += self.regr_loss(obj_br_regr, obj_gt_br_regr, obj_gt_mask)
        obj_regr_loss = self.regr_weight * obj_regr_loss

        loss = (sub_focal_loss + sub_pull_loss + sub_push_loss + sub_regr_loss) / len(sub_tl_heats) + \
               (obj_focal_loss + obj_pull_loss + obj_push_loss + obj_regr_loss) / len(obj_tl_heats) 
        return loss.unsqueeze(0)
