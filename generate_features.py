import argparse
import os
from tqdm import tqdm
import pickle

import torch
from torch.utils.collect_env import get_pretty_env_info
import torch.distributed as dist

from libs.config import cfg
from libs.dataset import make_data_loader
from libs.modeling.detector import build_detection_model
from libs.utils.checkpoint import ActionCheckpointer
from libs.dataset.datasets.evaluation import evaluate
from libs.utils.comm import get_rank, is_main_process, all_gather, synchronize, get_world_size
from libs.structures.memory_pool import MemoryPool
from libs.utils.comm import get_rank
from libs.utils.logger import setup_logger
import time

def normalize(video_id,prediction,dataset):
    TO_REMOVE = 1
    video_info = dataset.get_video_info(video_id)
    video_width = video_info["width"]
    video_height = video_info["height"]
    prediction = prediction.resize((video_width, video_height))
    prediction = prediction.convert("xyxy")
    boxes = prediction.bbox
    boxes[:, [2, 3]] += TO_REMOVE
    boxes[:, [0, 2]] /= video_width
    boxes[:, [1, 3]] /= video_height
    #print(boxes)
    boxes = torch.clamp(boxes, 0.0, 1.0)
    
    return boxes


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    video_ids = list(sorted(predictions.keys()))
    if len(video_ids) != video_ids[-1] + 1:
        logger = logging.getLogger("AlphAction.inference")
        logger.warning(
            "Number of videos that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in video_ids]
    return predictions

def gererate_features(data_loader, model, cfg, logger, is_train_loader):
    device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    
    outdir_p = './features/train_ext/actors_train_global/' if is_train_loader else './features/val_pred/actors_val_global/'
    outdir_o = './features/train_ext/objects_train_my/' if is_train_loader else './features/val_pred/objects_val_my/'
    
    results_dict = {}
    
    model.eval()
    for iteration, batch in enumerate(tqdm(data_loader)):
        slow_clips, fast_clips, boxes, objects, extras, video_ids = batch
        assert len(extras) == len(video_ids)

        slow_clips = slow_clips.to(device)
        fast_clips = fast_clips.to(device)
        boxes = [box.to(device) for box in boxes]
        objects = [None if (box is None) else box.to(device) for box in objects]

        with torch.no_grad():
            output,person_pooled,object_pooled = model(slow_clips, fast_clips, boxes, objects, extras)
            #print(person_pooled.shape,object_pooled.shape)
            output = [o.to(cpu_device) for o in output]
            person_pooled = person_pooled.to(cpu_device)
            object_pooled = object_pooled.to(cpu_device)
            boxes = [box.to(cpu_device) for box in boxes]
            #print(boxes)
            #print(person_pooled.shape)
            objects = [None if (box is None) else box.to(cpu_device) for box in objects]
            
            idx_p = 0
            idx_o = 0
            for i in range(len(extras)):
                movie_id = extras[i]['movie_id']
                timestamp = extras[i]['timestamp']     
                video_id = video_ids[i]
                
                ### Storage person features
                person_box = boxes[i]
                box_num = len(person_box)
                person_feature = person_pooled[idx_p:(idx_p+box_num)]
                idx_p += box_num
                box_normalize = normalize(video_id,person_box,data_loader.dataset)
                
                all_info = {}
                all_info['boxes'] = box_normalize
                all_info['feature'] = person_feature
                #all_info['labels']= person_box.get_field('labels')
                #print(all_info['boxes'].shape,all_info['feature'].shape,all_info['labels'].shape)
                
                person_dir = os.path.join(outdir_p,movie_id)
                if not os.path.exists(person_dir):
                    os.mkdir(person_dir)
                filename = os.path.join(person_dir,str(timestamp).zfill(4)+'.ft')
                f = open(filename,'wb')
                pickle.dump(all_info,f)
                
                ### Storage object features
                #object_box = objects[i]
                #box_num_ob = len(object_box)
                #object_feature = object_pooled[idx_o:idx_o+box_num_ob]
                #idx_o += box_num_ob
                #box_normalize = normalize(video_id,object_box,data_loader.dataset)
                
                #all_info = {}
                #all_info['boxes'] = box_normalize
                #all_info['feature'] = object_feature
                #print(all_info['boxes'].shape,all_info['feature'].shape)
                
                #object_dir = os.path.join(outdir_o,movie_id)
                #if not os.path.exists(object_dir):
                #    os.mkdir(object_dir)
                #filename = os.path.join(object_dir,str(timestamp).zfill(4)+'.ft')
                #f = open(filename,'wb')
                #pickle.dump(all_info,f)
                
        results_dict.update(
            {video_id: result for video_id, result in zip(video_ids, output)}
        )
        
    predictions = _accumulate_predictions_from_multiple_gpus(results_dict)
    
    if is_train_loader:
        return 0
    
    if cfg.OUTPUT_DIR:
        torch.save(predictions, os.path.join(cfg.OUTPUT_DIR, "predictions.pth"))

    return evaluate(
        dataset=data_loader.dataset,
        predictions=predictions,
        output_folder=cfg.OUTPUT_DIR,
    )
    

def main():    
    cfg.merge_from_file('./config_files/resnet50_4x16f_backbone.yaml')
    num_gpus = 1

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    logger = setup_logger("AlphAction", cfg.OUTPUT_DIR, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + get_pretty_env_info())

    # Build the model.
    model = build_detection_model(cfg)
    model = model.to("cuda")

    print(model)
    
    checkpointer = ActionCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR,save_to_disk=True)
    checkpointer.load(cfg.MODEL.WEIGHT)
    state = model.state_dict()

    data_loader_train = make_data_loader(cfg, is_train=False, is_distributed=False, flag= True)
    #data_loader_test  = make_data_loader(cfg, is_train=False, is_distributed=False, flag= False)
    
    #gererate_features(data_loader_test[0] , model, cfg, logger, is_train_loader=False)
    gererate_features(data_loader_train[0], model, cfg, logger, is_train_loader=True)
    
if __name__ == '__main__':
    main()