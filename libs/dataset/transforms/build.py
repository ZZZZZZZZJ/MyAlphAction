from . import video_transforms as T
#from . import video_transforms_rescale as T
from . import object_transforms as OT


def build_transforms(cfg, is_train=True):
    # build transforms for training of testing
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    flip_prob = 0
    slow_jitter = False
    crop_size = cfg.DATALOADER.CROP_SIZE

    frame_num = cfg.INPUT.FRAME_NUM
    sample_rate = cfg.INPUT.FRAME_SAMPLE_RATE

    to_bgr = cfg.INPUT.TO_BGR
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr=to_bgr
    )

    tau = cfg.INPUT.TAU
    alpha = cfg.INPUT.ALPHA

    transform = T.Compose(
        [
            T.TemporalCrop(frame_num, sample_rate),
            T.Resize(min_size, max_size),
            #T.Crop_Resize(crop_size,1.5,False),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
            T.SlowFastCrop(tau, alpha, slow_jitter),
        ]
    )

    return transform


def build_object_transforms(cfg, is_train=True):
    # build transforms for object boxes, should be kept consistent with video transforms.
    if is_train:
        flip_prob = 0.5
    else:
        flip_prob = 0

    transform = OT.Compose([
        OT.PickTop(cfg.AOG_STRUCTURE.MAX_OBJECT),
        OT.Resize(),
        OT.RandomHorizontalFlip(flip_prob)
    ])
    return transform
