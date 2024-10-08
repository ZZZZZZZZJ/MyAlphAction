import torch

from .roi_action_feature_extractor import make_roi_action_feature_extractor
from .roi_action_predictors import make_roi_action_predictor
from .inference import make_roi_action_post_processor
from libs.modeling.utils import prepare_pooled_feature
from libs.structures.bounding_box import BoxList

class ROIActionHead(torch.nn.Module):
    """
    Generic Action Head class.
    """

    def __init__(self, cfg, dim_in):
        super(ROIActionHead, self).__init__()
        self.feature_extractor = make_roi_action_feature_extractor(cfg, dim_in)
        self.predictor = make_roi_action_predictor(cfg, self.feature_extractor.dim_out)
        self.post_processor = make_roi_action_post_processor(cfg)
        self.test_ext = cfg.TEST.EXTEND_SCALE
        self.size = cfg.DATALOADER.CROP_SIZE

    def forward(self, slow_features, fast_features, boxes, objects=None, extras={}, pool = None, is_get_features = False, is_post_processing = False):
        # In training stage, boxes are from gt.
        # In testing stage, boxes are detected by human detector and proposals should be
        # enlarged boxes.
        if is_post_processing:
            proposals = None
        else:
            proposals = [box.extend(self.test_ext) for box in boxes]
        
        x, person_features = self.feature_extractor(slow_features, fast_features, proposals, objects, extras, pool, is_get_features, is_post_processing)
        
        pooled_feature = prepare_pooled_feature(person_features, boxes)
        
        if is_get_features:
            return None, pooled_feature

        action_logits = self.predictor(x)

        result = self.post_processor((action_logits,), boxes)
        return result, pooled_feature

    def c2_weight_mapping(self):
        weight_map = {}
        for name, m_child in self.named_children():
            if m_child.state_dict() and hasattr(m_child, "c2_weight_mapping"):
                child_map = m_child.c2_weight_mapping()
                for key, val in child_map.items():
                    new_key = name + '.' + key
                    weight_map[new_key] = val
        return weight_map


def build_roi_action_head(cfg, dim_in):
    return ROIActionHead(cfg, dim_in)
