import torch
import time
from torch import nn
from torch.nn import functional as F

from libs.modeling import registry
from libs.modeling.poolers import make_3d_pooler
from libs.modeling.roi_heads.action_head.IA_structure import make_ia_structure
from libs.modeling.roi_heads.action_head.AOG_structure import make_aog_structure
from libs.modeling.utils import cat, pad_sequence, prepare_pooled_feature
from libs.utils.IA_helper import has_object


@registry.ROI_ACTION_FEATURE_EXTRACTORS.register("2MLPFeatureExtractor")
class MLPFeatureExtractor(nn.Module):
    def __init__(self, config, dim_in):
        super(MLPFeatureExtractor, self).__init__()
        self.config = config
        head_cfg = config.MODEL.ROI_ACTION_HEAD

        self.pooler = make_3d_pooler(head_cfg)

        resolution = head_cfg.POOLER_RESOLUTION

        self.max_pooler = nn.MaxPool3d((1, resolution, resolution))
        if config.AOG_STRUCTURE.ACTIVE == True:
            self.aog_structure = make_aog_structure(config, dim_in)

        representation_size = head_cfg.MLP_HEAD_DIM

        fc1_dim_in = dim_in
        if config.AOG_STRUCTURE.ACTIVE == True and config.AOG_STRUCTURE.FUSION == 'concat':
            fc1_dim_in = dim_in + config.AOG_STRUCTURE.DIM_OUT

        #self.fc1_m = nn.Linear(fc1_dim_in, representation_size)
        self.fc1 = nn.Linear(fc1_dim_in, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

        #for l in [self.fc1_m, self.fc2]:
        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

        self.dim_out = representation_size

    def roi_pooling(self, slow_features, fast_features, proposals):
        if slow_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_features = slow_features.mean(dim=2, keepdim=True)
            slow_x = self.pooler(slow_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                slow_x = slow_x.mean(dim=2, keepdim=True)
            x = slow_x
        if fast_features is not None:
            if self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_features = fast_features.mean(dim=2, keepdim=True)
            fast_x = self.pooler(fast_features, proposals)
            if not self.config.MODEL.ROI_ACTION_HEAD.MEAN_BEFORE_POOLER:
                fast_x = fast_x.mean(dim=2, keepdim=True)
            x = fast_x

        if slow_features is not None and fast_features is not None:
            x = torch.cat([slow_x, fast_x], dim=1)
        return x

    def max_pooling_zero_safe(self, x):
        if x.size(0) == 0:
            _, c, t, h, w = x.size()
            res = self.config.MODEL.ROI_ACTION_HEAD.POOLER_RESOLUTION
            x = torch.zeros((0, c, 1, h - res + 1, w - res + 1), device=x.device)
        else:
            x = self.max_pooler(x)
        return x

    def forward(self, slow_features, fast_features, proposals, objects=None, extras={}, part_forward=-1):
        aog_active = hasattr(self, "aog_structure")     

        x = self.roi_pooling(slow_features, fast_features, proposals)

        person_pooled = self.max_pooler(x)

        #if aog_active:
        object_pooled = self.roi_pooling(slow_features, fast_features, objects)
        object_pooled = self.max_pooling_zero_safe(object_pooled)
        #else:
        #    object_pooled = None        

        x_after = person_pooled

        if aog_active:
            aog_feature = self.aog_structure(person_pooled, proposals, object_pooled, objects )
            x_after = self.fusion(x_after, aog_feature, self.config.AOG_STRUCTURE.FUSION)

        x_after = x_after.view(x_after.size(0), -1)

        #x_after = F.relu(self.fc1_m(x_after))
        x_after = F.relu(self.fc1(x_after))
        x_after = F.relu(self.fc2(x_after))

        return x_after, person_pooled, object_pooled

    def check_fetch_mem_feature(self, movie_cache, mem_ind, max_num):
        if mem_ind not in movie_cache:
            return None
        box_list = movie_cache[mem_ind]
        return self.sample_mem_feature(box_list, max_num)

    def sample_mem_feature(self, box_list, max_num):
        if len(box_list) > max_num:
            idx = torch.randperm(len(box_list))[:max_num]
            return box_list[idx].to("cuda")
        else:
            return box_list.to("cuda")

    def fusion(self, x, out, type="add"):
        if type == "add":
            return x + out
        elif type == "concat":
            return torch.cat([x, out], dim=1)
        else:
            raise NotImplementedError


def make_roi_action_feature_extractor(cfg, dim_in):
    func = registry.ROI_ACTION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_ACTION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, dim_in)
