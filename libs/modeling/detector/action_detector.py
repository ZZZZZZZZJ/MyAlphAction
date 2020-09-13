from torch import nn
import torch

from ..backbone import build_backbone
from ..roi_heads.roi_heads_3d import build_3d_roi_heads


class ActionDetector(nn.Module):
    def __init__(self, cfg):
        super(ActionDetector, self).__init__()
        self.backbone = build_backbone(cfg)
        self.roi_heads = build_3d_roi_heads(cfg, self.backbone.dim_out)

    def forward(self, slow_video, fast_video, boxes, objects=None, extras={}, pool =None, is_get_features = False, is_post_processing=False):
        # part_forward is used to split this model into two parts.
        # if part_forward<0, just use it as a single model
        # if part_forward=0, use this model to extract pooled feature(person and object, no memory features).
        # if part_forward=1, use the ia structure to aggregate interactions and give final result.
        # implemented in roi_heads
        if is_post_processing:
            
            result, person_features = self.roi_heads(None, None, boxes, None, extras, pool, is_get_features, is_post_processing)
            
        else:
            slow_features, fast_features = self.backbone(slow_video, fast_video)

            #result, detector_losses, detector_metrics = self.roi_heads(slow_features, fast_features, boxes, objects, extras, ori_boxes)
            result, person_features = self.roi_heads(slow_features, fast_features, boxes, objects, extras, pool, is_get_features)

        return result, person_features

    def c2_weight_mapping(self):
        if not hasattr(self, "c2_mapping"):
            weight_map = {}
            for name, m_child in self.named_children():
                if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                    child_map = m_child.c2_weight_mapping()
                    for key, val in child_map.items():
                        new_key = name + '.' + key
                        weight_map[new_key] = val
            self.c2_mapping = weight_map
        return self.c2_mapping

def build_detection_model(cfg):
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"
    
    cur_device = torch.cuda.current_device()
    model = ActionDetector(cfg)
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device, #find_unused_parameters=True
        )
        
    return model