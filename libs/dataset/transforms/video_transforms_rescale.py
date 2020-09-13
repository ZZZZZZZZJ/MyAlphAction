import torch
import random
import numpy as np
import cv2
from libs.structures.bounding_box import BoxList

cv2.setNumThreads(0)


class Compose(object):
    # Compose different kinds of video transforms into one.
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, videos, target):
        transform_randoms = {}
        ori_boxes = {}
        for t in self.transforms:
            videos, target, ori_boxes, transform_randoms = t(videos, target, ori_boxes, transform_randoms)
        return videos, target, ori_boxes, transform_randoms

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class TemporalCrop(object):
    def __init__(self, frame_num, sample_rate, temporal_jitter=0):
        self.frame_num = frame_num
        self.sample_rate = sample_rate
        self.temporal_jitter = temporal_jitter

    def __call__(self, clip, target, ori_boxes, transform_randoms):
        # crop the input frames from raw clip
        raw_frame_num = clip.shape[0]

        # determine frame start based on frame_num, sample_rate and the jitter shift.
        frame_start = (raw_frame_num - self.frame_num * self.sample_rate) // 2 + (self.sample_rate - 1) // 2 + self.temporal_jitter
        idx = np.arange(frame_start, frame_start + self.frame_num * self.sample_rate, self.sample_rate)
        idx = np.clip(idx, 0, raw_frame_num - 1)

        clip = clip[idx]
        return clip, target, ori_boxes, transform_randoms
    
class Crop_Resize(object):
    def __init__(self, size, expand_rate, is_train=True):
        self.size = size
        self.expand_rate = expand_rate
        self.is_train = is_train
        
    def __call__(self, clip, target, ori_boxes, transform_randoms):
        # Crop and resize
        if target is None:
            print('box no found')
            return clip, target, transform_randoms
        
        origin_size = clip.shape[1:3]
        ori_boxes = target
        ex_target = target.resize(origin_size)
        
        # Extend box_size
        if self.is_train:
            rate = (np.random.rand(1)*(self.expand_rate-1)+1)[0]
        else:
            rate = self.expand_rate
        ex_target = ex_target.extend((rate,rate))
        
        # Crop
        #target = target.convert("xyxy")
        bbox = ex_target.bbox
        clip_list = []
        box_list = []
        for i in range(bbox.shape[0]):
            TO_REMOVE = 1
            xmin, ymin, xmax, ymax = bbox[i]
            box = bbox[i]
            xmin, ymin, xmax, ymax = xmin.int(), ymin.int(), xmax.int(), ymax.int()
            w, h = xmax - xmin, ymax - ymin
            c = max(w,h)
            if w < c:
                padw = (c - w) / 2
                xmin = (xmin - int(padw)).clamp(min=0, max=origin_size[0])
                xmax = (xmin + c).clamp(min=0, max=origin_size[0])
            else:
                padh = (c - h) / 2
                ymin = (ymin - int(padh)).clamp(min=0, max=origin_size[1])
                ymax = (ymin + c).clamp(min=0, max=origin_size[1])
            #assert (xmax-xmin) == (ymax-ymin)
            clip_frames = clip[:,int(xmin):int(xmax),int(ymin):int(ymax),:]
            # c*c
            cropped_xmin = (box[0:1] - xmin).clamp(min=0, max=c)
            cropped_ymin = (box[1:2] - ymin).clamp(min=0, max=c)
            cropped_xmax = (box[2:3] - xmin).clamp(min=0, max=c)
            cropped_ymax = (box[3:4] - ymin).clamp(min=0, max=c)
            cropped_box = torch.cat(
                (cropped_xmin, cropped_ymin, cropped_xmax, cropped_ymax), dim=-1
            )
            crop_bbox = BoxList(cropped_box.unsqueeze(0), (c, c), mode="xyxy")
            box_list.append(crop_bbox)
            #resize
            size = self.size
            clip_new = np.zeros((clip.shape[0], size[1], size[0], clip.shape[3]), dtype=np.uint8)
            for i in range(clip.shape[0]):
                cv2.resize(clip[i], size, clip_new[i])
            clip_list.append(clip_new)
        new_clip = np.stack(clip_list,axis=0)
        
        transform_randoms["Resize"] = self.size
        
        return new_clip, box_list, ori_boxes, transform_randoms
        


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        # Calculate output size according to min_size, max_size.
        h, w = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (ow, oh)

    def __call__(self, clip, target, transform_randoms):
        # Input clip should be [TxHxWxC](uint8) ndarray.
        size = self.get_size(clip.shape[1:3])
        clip_new = np.zeros((clip.shape[0], size[1], size[0], clip.shape[3]), dtype=np.uint8)
        for i in range(clip.shape[0]):
            cv2.resize(clip[i], size, clip_new[i])
        if target is not None:
            target = target.resize(size)
        # Store the size for object box transforms.
        transform_randoms["Resize"] = size
        return clip_new, target, transform_randoms


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, clip, target, ori_boxes, transform_randoms):
        # Input clip should be [TxHxWxC] ndarray.(uint8)
        flip_random = random.random()
        if flip_random < self.prob:
            #clip = np.flip(clip, 2)
            #modify it to [B,T,H,W,C]
            clip = np.flip(clip, 3)
            if target is not None:
                target = target.transpose(0)
            if ori_boxes is not None:
                ori_boxes = ori_boxes.transpose(0)

        # Store the random variable for object box transforms
        transform_randoms["Flip"] = flip_random
        return clip, target, ori_boxes, transform_randoms


class ToTensor(object):
    def __call__(self, clip, target, ori_boxes, transform_randoms):
        # Input clip should be [TxHxWxC] ndarray.
        # Convert to [CxTxHxW] tensor.
        
        #return torch.from_numpy(clip.transpose((3, 0, 1, 2)).astype(np.float32)), target, transform_randoms
        #modify it to [B,T,H,W,C]
        # convert to [B,C,T,H,W]
        #print(clip.shape)
        return torch.from_numpy(clip.transpose((0, 4, 1, 2, 3)).astype(np.float32)), target, ori_boxes, transform_randoms


class Normalize(object):
    def __init__(self, mean, std, to_bgr=False):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def video_normalize(self, tensor, mean, std):
        # Copied from torchvision.transforms.functional.normalize but remove the type check of tensor.
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor

    def __call__(self, clip, target, ori_boxes, transform_randoms):
        if self.to_bgr:
            #clip = clip[[2, 1, 0]]
            #modify it to [B,C,T,H,W]
            clip = clip[:,[2,1,0]]
        # normalize: (x-mean)/std
        clip = self.video_normalize(clip, self.mean, self.std)
        return clip, target, ori_boxes, transform_randoms


class SlowFastCrop(object):
    # Class used to split frames for slow pathway and fast pathway.
    def __init__(self, tau, alpha, slow_jitter=False):
        self.tau = tau
        self.alpha = alpha
        self.slow_jitter = slow_jitter

    def __call__(self, clip, target, ori_boxes, transform_randoms):
        if self.slow_jitter:
            # if jitter, random choose a start
            slow_start = random.randint(0, self.tau - 1)
        else:
            # if no jitter, select the middle
            slow_start = (self.tau - 1) // 2
        #slow_clip = clip[:, slow_start::self.tau, :, :]
        slow_clip = clip[:, :, slow_start::self.tau, :, :]

        fast_stride = self.tau // self.alpha
        fast_start = (fast_stride - 1) // 2
        #fast_clip = clip[:, fast_start::fast_stride, :, :]
        fast_clip = clip[:, :, fast_start::fast_stride, :, :]

        return [slow_clip, fast_clip], target, ori_boxes, transform_randoms

class Identity(object):
    # Return what is received. Do nothing.
    def __init__(self):
        pass

    def __call__(self, clip, target, ori_boxes, tranform_randoms):
        return clip, target, ori_boxes, tranform_randoms
