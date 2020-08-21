from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import math
import torch
import torch.nn as nn
from libs.modeling import registry    
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

@registry.INTERACTION_AGGREGATION_STRUCTURES.register("RN")
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
        
        if structure_cfg.USE_ZERO_INIT_CONV:
            out_init = 0
        else:
            out_init = init_std
            
        self.Iblock_object = Relation_block(structure_cfg, dim_person=self.dim_inner, dim_others=self.dim_inner, dropout_rate= structure_cfg.DROPOUT)
        self.Iblock_person = Relation_block(structure_cfg, dim_person=self.dim_inner, dim_others=self.dim_inner, dropout_rate= structure_cfg.DROPOUT)          
            
    def _reduce_dim(self, persons, objs):
        persons_features = persons.squeeze(4)
        objs_features = objs.squeeze(4)
        
        persons_red = self.p_reduce_conv(persons_features)
        act_persons_red = self.pr_reduce_conv(persons_features)
        objs_red = self.o_reduce_conv(objs_features)
        
        return persons_red, objs_red, act_persons_red
            

    def forward(self, person_feature, person_boxes, obj_feature, object_boxes):

        persons_red,objs_red,act_persons_red = self._reduce_dim(person_feature, obj_feature)

        objs_interact = self.Iblock_object(persons_red, objs_red, person_boxes, object_boxes, False)
        persons_interact = self.Iblock_person(persons_red, act_persons_red, person_boxes, person_boxes, True)
        
        x = torch.cat((objs_interact,persons_interact),dim=1)
        
        x = x.unsqueeze(4)
        
        return x


def init_layer(layer, init_std, bias):
    if init_std == 0:
        nn.init.constant_(layer.weight, 0)
    else:
        nn.init.normal_(layer.weight, std=init_std)
    if bias:
        nn.init.constant_(layer.bias, 0)

def make_aog_structure(cfg, dim_in):
    func = registry.INTERACTION_AGGREGATION_STRUCTURES[
        cfg.AOG_STRUCTURE.STRUCTURE
    ]
    print(cfg.AOG_STRUCTURE.STRUCTURE)
    return func(dim_in, cfg.AOG_STRUCTURE.DIM_OUT, cfg.AOG_STRUCTURE)
