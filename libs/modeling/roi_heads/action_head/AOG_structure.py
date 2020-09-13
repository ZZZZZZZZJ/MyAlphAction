from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import torch
import torch.nn as nn
import time

class Relation_block(torch.nn.Module):
    def __init__(self, cfg, dim_person, dim_others, dropout_rate):
        super(Relation_block, self).__init__()
        
        self.softmax = nn.Softmax(dim = 1)
        
        self.MLP = nn.Linear(dim_person + dim_others, dim_person, bias=True)
        
    def forward(self, person_features, other_features, person_boxes, other_boxes, is_person):
        f_num = len(person_boxes)
        res = []
        idx_p = 0
        idx_o = 0
        for i in range(f_num):
            np = len(person_boxes[i])
            no = len(other_boxes[i])
            #print(np, no)
            person = person_features[idx_p:(idx_p+np)].squeeze(3).squeeze(2)
            other  = other_features[idx_o:(idx_o+no)].squeeze(3).squeeze(2)
            idx_p += np
            idx_o += no
            if no == 0:
                x = torch.zeros(person.shape).cuda()
            elif no == 1 and is_person:
                x = torch.zeros(person.shape).cuda()
            else:
                persons = torch.repeat_interleave(person,no,dim=0)
                others = torch.cat([other for i in range(np)])
                x = torch.cat((persons,others),dim=1)
                if is_person:
                    indices = torch.linspace(0,np*no-1,np*no).cuda().long()
                    indices = indices[indices%(no+1)!=0]
                    x = torch.index_select(x, 0, indices)
                x = self.MLP(x)
                if is_person:
                    x = x.contiguous().view(np,no-1,-1)
                else:
                    x = x.contiguous().view(np,no,-1)
                x = torch.max(x, 1).values
            
            res.append(x)
        res = torch.cat(res,dim=0)
        return res.unsqueeze(2).unsqueeze(3)
    
class Relation_block_batch(torch.nn.Module):
    def __init__(self, cfg, dim_person, dim_others, dim_inner, dropout_rate):
        super(Relation_block_batch, self).__init__()
        
        self.softmax = nn.Softmax(dim = 1)
        
        self.dim_inner = dim_inner
        
        #bias = not cfg.structure_cfg.NO_BIAS
        #conv_init_std = cfg.structure_cfg.CONV_INIT_STD

        #self.p_conv = nn.Conv2d(dim_person, self.dim_inner, 1, bias)  # reduce person query
        #init_layer(self.p_conv, conv_init_std, bias)
        
        #self.o_conv = nn.Conv2d(dim_others, self.dim_inner, 1, bias)  # reduce person query
        #init_layer(self.o_conv, conv_init_std, bias)
        
        self.MLP = nn.Linear(dim_inner + dim_inner, dim_inner, bias=True)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, person_features, other_features, person_boxes, other_boxes, is_person):
        # (n, c, w, h), (n, max_others, c, w, h)
        n, c, w, h = person_features.shape
        n, max_others, c, w, h = other_features.shape
        
        #person_t = self.p_conv(person_features)
        #other_t = unfuse_batch_num(self.o_conv(fuse_batch_num(other_features)), n, max_others)
        
        key = person_features.unsqueeze(1).repeat(1,max_others,1,1,1)
        x = torch.cat((key, other_features),dim=2)
        x = fuse_batch_num(x).squeeze(3).squeeze(2)
        
        #x = self.dropout(x)
        
        x = self.MLP(x).unsqueeze(2).unsqueeze(3)
        x = unfuse_batch_num(x, n, max_others)
        x = torch.max(x, 1).values
        
        #x = person_features + x
        #print(x.shape)
        return x

class AOGStructure(nn.Module):
    def __init__(self, dim_person, dim_out, structure_cfg):
        super(AOGStructure, self).__init__()
        
        self.dim_inner = structure_cfg.DIM_INNER
        self.dim_out = structure_cfg.DIM_OUT
        self.max_person = structure_cfg.MAX_PERSON
        self.max_object = structure_cfg.MAX_OBJECT
        
        bias = not structure_cfg.NO_BIAS
        conv_init_std = structure_cfg.CONV_INIT_STD

        self.p_reduce_conv = nn.Conv2d(dim_person, self.dim_inner, 1, bias)  # reduce person query
        init_layer(self.p_reduce_conv, conv_init_std, bias)
        
        self.pr_reduce_conv = nn.Conv2d(dim_person, self.dim_inner, 1, bias)  # reduce person query
        init_layer(self.pr_reduce_conv, conv_init_std, bias)
        
        self.o_reduce_conv = nn.Conv2d(dim_person, self.dim_inner, 1, bias)
        init_layer(self.o_reduce_conv, conv_init_std, bias)
        
        self.dropout = nn.Dropout(structure_cfg.DROPOUT)
        self.reduce_dropout = nn.Dropout(structure_cfg.REDUCE_DROPOUT)
        
        if structure_cfg.USE_ZERO_INIT_CONV:
            out_init = 0
        else:
            out_init = init_std
            
        #self.Iblock_object = Relation_block(structure_cfg, dim_person=self.dim_inner, dim_others=self.dim_inner, dropout_rate= structure_cfg.DROPOUT)
        #self.Iblock_person = Relation_block(structure_cfg, dim_person=self.dim_inner, dim_others=self.dim_inner, dropout_rate= structure_cfg.DROPOUT)          
        self.Iblock_object = Relation_block_batch(structure_cfg, dim_person=self.dim_inner, dim_others=self.dim_inner, dim_inner = self.dim_inner, dropout_rate= structure_cfg.DROPOUT)
        self.Iblock_person = Relation_block_batch(structure_cfg, dim_person=self.dim_inner, dim_others=self.dim_inner, dim_inner = self.dim_inner, dropout_rate= structure_cfg.DROPOUT)  
            
    def _aggregrate_features(self, feat):
        return torch.cat(feat,dim=0)
    
    def _reduce_dim(self, persons, objs, person_boxes, object_boxes):
        person_features = persons.squeeze(4) 
        n, c, w, h = person_features.shape
        p_red = self.p_reduce_conv(person_features)
        p_red = self.reduce_dropout(p_red)
        
        # reduce acting person features
        pr_red = self.pr_reduce_conv(person_features)
        pr_red = separate_roi_per_person(person_boxes, pr_red, person_boxes, self.max_person, )
        pr_red = self.reduce_dropout(pr_red)
        
        # reduce object features
        object_features = objs.squeeze(4)
        object_feat = separate_roi_per_person(person_boxes, object_features, object_boxes,
                                                 self.max_object)
        object_feat = fuse_batch_num(object_feat)
        o_red = self.o_reduce_conv(object_feat)
        o_red = unfuse_batch_num(o_red, n, self.max_object)
        o_red = self.reduce_dropout(o_red)
        
        return p_red, o_red, pr_red
            

    def forward(self, person_feature, person_boxes, obj_feature, object_boxes):

        persons_red,objs_red,act_persons_red = self._reduce_dim(person_feature, obj_feature, person_boxes, object_boxes)
        #print(persons_red.shape, objs_red.shape, act_persons_red.shape)

        objs_interact = self.Iblock_object(persons_red, objs_red, person_boxes, object_boxes, False)
        persons_interact = self.Iblock_person(persons_red, act_persons_red, person_boxes, person_boxes, True)
        
        x = torch.cat((objs_interact,persons_interact),dim=1)
        
        x = x.unsqueeze(4)
        
        return x

def separate_roi_per_person(proposals, things, other_proposals, max_things):
    """
    :param things: [n2, c, h, w]
    :param proposals:
    :param max_things:
    :return [n, max_other, c, h, w]
    """
    res = []
    _, c, h, w = things.size()
    device = things.device
    index = 0
    for i, (person_box, other_box) in enumerate(zip(proposals, other_proposals)):
        person_num = len(person_box)
        other_num = len(other_box)
        tmp = torch.zeros((person_num, max_things, c, h, w), device=device)
        if other_num > max_things:
            idx = torch.randperm(other_num)[:max_things]
            tmp[:, :max_things] = things[index:index + other_num][idx]
        else:
            tmp[:, :other_num] = things[index:index + other_num]

        res.append(tmp)
        index += other_num
    features = torch.cat(res, dim=0)
    return features

def fuse_batch_num(things):
    n, number, c, h, w = things.size()
    return things.contiguous().view(-1, c, h, w)

def unfuse_batch_num(things, batch_size, num):
    assert things.size(0) == batch_size * num, "dimension should matches"
    _, c, h, w = things.size()
    return things.contiguous().view(batch_size, num, c, h, w)

def init_layer(layer, init_std, bias):
    if init_std == 0:
        nn.init.constant_(layer.weight, 0)
    else:
        nn.init.normal_(layer.weight, std=init_std)
    if bias:
        nn.init.constant_(layer.bias, 0)

def make_aog_structure(cfg, dim_in):
    return AOGStructure(dim_in, cfg.AOG_STRUCTURE.DIM_OUT, cfg.AOG_STRUCTURE)
