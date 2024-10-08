import torch

from .action_head.action_head import build_roi_action_head


class Combined3dROIHeads(torch.nn.ModuleDict):
    def __init__(self, cfg, heads):
        super(Combined3dROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()

    def forward(self, slow_features, fast_features, boxes, objects=None, extras={}, pool = None, is_get_features = False, is_post_processing = False):
        result, person_features = self.action(slow_features, fast_features, boxes, objects, extras, pool, is_get_features, is_post_processing)

        return result, person_features

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict() and hasattr(m_child,"c2_weight_mapping"):
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    weight_map[new_key] = val
        return weight_map


def build_3d_roi_heads(cfg, dim_in):
    roi_heads = []
    roi_heads.append(("action", build_roi_action_head(cfg, dim_in)))

    if roi_heads:
        roi_heads = Combined3dROIHeads(cfg, roi_heads)

    return roi_heads
