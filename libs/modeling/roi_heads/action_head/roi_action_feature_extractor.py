import torch
import time
from torch import nn
from torch.nn import functional as F

from libs.modeling import registry
from libs.modeling.poolers import make_3d_pooler
from libs.modeling.roi_heads.action_head.IA_structure import make_ia_structure
from libs.modeling.roi_heads.action_head.AOG_structure import make_aog_structure, Relation_block_batch, separate_roi_per_person, fuse_batch_num, unfuse_batch_num, init_layer
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
        
        self.memory = config.AOG_STRUCTURE.MEMORY
        #self.memory = False

        self.max_pooler = nn.MaxPool3d((1, resolution, resolution))
        if config.AOG_STRUCTURE.ACTIVE == True:
            self.aog_structure = make_aog_structure(config, dim_in)
            if self.memory == True:
                self.mem_block = Memory_block(config)

        representation_size = head_cfg.MLP_HEAD_DIM

        fc1_dim_in = dim_in
        if config.AOG_STRUCTURE.ACTIVE == True and config.AOG_STRUCTURE.FUSION == 'concat':
            fc1_dim_in = dim_in + config.AOG_STRUCTURE.DIM_OUT
        fc2_dim_in = representation_size
        if config.AOG_STRUCTURE.ACTIVE == True and self.memory == True:
            fc2_dim_in = representation_size + config.AOG_STRUCTURE.DIM_INNER

        self.fc1_m = nn.Linear(fc1_dim_in, representation_size)
        #self.fc1 = nn.Linear(fc1_dim_in, representation_size)
        #self.fc2 = nn.Linear(representation_size, representation_size)
        self.fc2_m = nn.Linear(fc2_dim_in, representation_size)

        for l in [self.fc1_m, self.fc2_m]:
        #for l in [self.fc1, self.fc2]:
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
    
    def _get_current_features(self, pool, extras):
        feature_list = []
        box_list = []
        movie_ids = [e["movie_id"] for e in extras]
        timestamps = [e["timestamp"] for e in extras]
        for movie_id, timestamp in zip(movie_ids, timestamps):
            cache_cur_box = pool[movie_id][timestamp].to("cuda")
            feature_list.append(cache_cur_box.get_field("pooled_feature").to("cuda"))
            box_list.append(cache_cur_box.bbox)
        return torch.cat(feature_list,dim=0), box_list

    def forward(self, slow_features, fast_features, proposals, objects=None, extras={}, pool=None, is_get_features = False, is_post_processing = False):
        if is_post_processing:
            x, person_boxes = self._get_current_features(pool, extras)
            person_features = x
            x_memory = self.mem_block(x, person_boxes, pool, extras)
            x_after = torch.cat((x,x_memory),dim=1)
            x_after = F.relu(self.fc2_m(x_after))
            return x_after, person_features
        
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

        x_after = F.relu(self.fc1_m(x_after))
        person_features = x_after
        if is_get_features:
            return None, person_features
        if hasattr(self, "mem_block"):
            x_memory = self.mem_block(x_after, proposals, pool, extras)
            x_after = torch.cat((x_after,x_memory),dim=1)

        #x_after = F.relu(self.fc1(x_after))
        #x_after = F.relu(self.fc2(x_after))
        x_after = F.relu(self.fc2_m(x_after))

        return x_after, person_features
    

    def fusion(self, x, out, type="add"):
        if type == "add":
            return x + out
        elif type == "concat":
            return torch.cat([x, out], dim=1)
        else:
            raise NotImplementedError

class Memory_block(nn.Module):
    def __init__(self, cfg):
        super(Memory_block, self).__init__()
        self.cfg = cfg
        self.mem_len = (30,30)
        self.max_boxes = 5
        self.mem_rate = 1
        self.max_memory = self.mem_len[0] * self.max_boxes * 2
        self.dim_inner = cfg.AOG_STRUCTURE.DIM_INNER
        self.dim_representation = cfg.MODEL.ROI_ACTION_HEAD.MLP_HEAD_DIM
        self.dropout = cfg.AOG_STRUCTURE.DROPOUT
        self.reduce_dropout = nn.Dropout(cfg.AOG_STRUCTURE.REDUCE_DROPOUT)
        
        bias = not cfg.AOG_STRUCTURE.NO_BIAS
        conv_init_std = cfg.AOG_STRUCTURE.CONV_INIT_STD
        
        self.Iblock_memory = Relation_block_batch(cfg, dim_person=self.dim_inner, dim_others=self.dim_inner, dim_inner = self.dim_inner, dropout_rate= self.dropout)
        
        self.mp_reduce_conv = nn.Conv2d(self.dim_representation, self.dim_inner, 1, bias)
        init_layer(self.mp_reduce_conv, conv_init_std, bias)
        
        self.pos_embed = nn.Embedding(self.mem_len[0]*2+1, 16, padding_idx=0)
        
        self.m_reduce_conv = nn.Conv2d(self.dim_representation+16, self.dim_inner, 1, bias)
        init_layer(self.m_reduce_conv, conv_init_std, bias)
        
    def _reduce_dim_m(self, person_features, mem_features, person_boxes, mem_boxes):
        person_features = person_features.unsqueeze(2).unsqueeze(3)
        n, c, w, h = person_features.shape
        p_red = self.mp_reduce_conv(person_features)
        p_red = self.reduce_dropout(p_red)
        
        mem_features = self._aggregrate_features(mem_features).unsqueeze(2).unsqueeze(3)
        mem_features = separate_roi_per_person(person_boxes, mem_features, mem_boxes, self.max_memory)
        mem_features = fuse_batch_num(mem_features)
        m_red = self.m_reduce_conv(mem_features)
        m_red = unfuse_batch_num(m_red, n, self.max_memory)
        m_red = self.reduce_dropout(m_red)
        
        return p_red, m_red
        
    def forward(self, x, person_boxes, pool, extras):
        memory_feat,memory_box = self._get_memory(x.device, pool, extras)
        x_red, m_red = self._reduce_dim_m(x, memory_feat, person_boxes, memory_box)
        hm_int = self.Iblock_memory(x_red, m_red, person_boxes, memory_box, False).squeeze(3).squeeze(2)
        
        return hm_int
        
    def _get_memory(self, device, pool, extras):
        before, after = self.mem_len
        max_boxes = self.max_boxes
        mem_rate = self.mem_rate
        feature_list = []
        box_list = []
        movie_ids = [e["movie_id"] for e in extras]
        timestamps = [e["timestamp"] for e in extras]
        for movie_id, timestamp in zip(movie_ids, timestamps):
            before_inds = range(timestamp - before * mem_rate, timestamp, mem_rate)
            after_inds = range(timestamp + mem_rate, timestamp + (after + 1) * mem_rate, mem_rate)
            cache_cur_mov = pool[movie_id]
            mem_box_list_before = [self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes)
                                   for mem_ind in before_inds]
            mem_box_list_after = [self.check_fetch_mem_feature(cache_cur_mov, mem_ind, max_boxes)
                                  for mem_ind in after_inds]
            mem_box_list = mem_box_list_before + mem_box_list_after
            
            mem_feature_list = []
            mem_feature_list += [box_list.get_field("pooled_feature")
                                 if box_list is not None
                                 else torch.zeros(0, self.dim_representation+16, dtype=torch.float32, device="cuda")
                                 for box_list in mem_box_list]
            mem_pos_list = []
            mem_pos_list += [box_list.bbox
                             if box_list is not None
                             else torch.zeros(0, 4, dtype=torch.float32, device="cuda")
                             for box_list in mem_box_list]
            
            mem_feat = self._aggregrate_mem_features(mem_feature_list)
            mem_feat = mem_feat.to(device)
            feature_list.append(mem_feat)
            
            mem_box = self._aggregrate_features(mem_pos_list)
            mem_box = mem_box.to(device)
            #print(mem_feat.shape, mem_box.shape)
            box_list.append(mem_box)
        return feature_list, box_list
    
    def _aggregrate_features(self, feat):
        return torch.cat(feat,dim=0)
    
    def _aggregrate_mem_features(self, feat_list):
        features = []
        for i, feat in enumerate(feat_list):
            n,d = feat.shape
            if n == 0:
                continue
            t_pos = torch.LongTensor([i]).to("cuda")
            t_emb = self.pos_embed(t_pos).repeat(n,1)
            features.append(torch.cat((feat,t_emb),dim=1))

        if len(features) > 0:
            return torch.cat(features,dim=0)
        return torch.zeros(0,self.dim_representation+16, dtype=torch.float32, device="cuda")
    
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
    

def make_roi_action_feature_extractor(cfg, dim_in):
    func = registry.ROI_ACTION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_ACTION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, dim_in)
