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

def train_epoch(train_loader, model, cur_epoch, cfg, checkpointer, logger, local_rank):
    learning_rate = 4 * 1e-5
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    device = torch.device("cuda")
    
    extra_args = {}
    
    model.train()
    
    for iteration, batch in enumerate(train_loader, **extra_args):
        slow_clips, fast_clips, boxes, objects, extras, video_ids = batch
        slow_clips = slow_clips.to(device)
        fast_clips = fast_clips.to(device)
        boxes = [box.to(device) for box in boxes]
        objects = [None if (box is None) else box.to(device) for box in objects]

        output = model(slow_clips, fast_clips, boxes, objects, extras)
        output = output[2]

        scores = []
        labels = []
        for i in range(len(output)):
            scores.append(output[i].get_field('scores'))
            labels.append(boxes[i].get_field('labels'))
        scores = torch.cat(scores,dim=0)
        labels = torch.cat(labels,dim=0)
        
        loss_fun = torch.nn.BCELoss(reduction="mean")
        # Compute the loss.
        loss = loss_fun(scores, labels.type(torch.float))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % 20 == 0:
            logger.info("GPU: " + str(local_rank) +  ", epoch: " + str(cur_epoch) + ", iter: " + str(iteration) + "/" + str(len(train_loader)) + ", lr: " + str(optimizer.param_groups[0]["lr"]) + ", class_loss: " + str(loss.item()) )
    checkpointer.save('AOG'+str(cur_epoch))

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

def eval_epoch(eval_loader, model, cur_epoch, cfg, checkpointer, logger):
    device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    
    extra_args = {}
    
    results_dict = {}
    
    model.eval()
    for iteration, batch in enumerate(tqdm(eval_loader,**extra_args)):
        slow_clips, fast_clips, boxes, objects, extras, video_ids = batch

        slow_clips = slow_clips.to(device)
        fast_clips = fast_clips.to(device)
        boxes = [box.to(device) for box in boxes]
        objects = [None if (box is None) else box.to(device) for box in objects]

        with torch.no_grad():
            output = model(slow_clips, fast_clips, boxes, objects, extras)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {video_id: result for video_id, result in zip(video_ids, output)}
        )
        
    synchronize()
    predictions = _accumulate_predictions_from_multiple_gpus(results_dict)
    
    if cfg.OUTPUT_DIR:
        torch.save(predictions, os.path.join(cfg.OUTPUT_DIR, "predictions.pth"))

    return evaluate(
        dataset=eval_loader.dataset,
        predictions=predictions,
        output_folder=cfg.OUTPUT_DIR,
    )
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    
    cfg.merge_from_file('./config_files/resnet50_4x16f_AOG.yaml')
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

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
    if distributed:
        print('local_rank',args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], 
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=True
                                                      )

    print(model)

    checkpointer = ActionCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR,save_to_disk=True)
    checkpointer.load(cfg.MODEL.WEIGHT)
    state = model.state_dict()

    data_loader_train = make_data_loader(cfg, is_train=True, is_distributed=distributed)
    data_loader_test = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    start_epoch = 12
    for cur_epoch in range(start_epoch,50):
        train_epoch(data_loader_train,model,cur_epoch,cfg,checkpointer,logger, args.local_rank)
        if cur_epoch % 1 == 0:
            eval_epoch(data_loader_test[0],model,cur_epoch,cfg,checkpointer,logger)
        cur_epoch += 1
            
    
if __name__ == '__main__':
    main()

#f = open(os.path.join(save_dir,'train_features.dat'), 'wb')
#pickle.dump(batch_info_list,f)
#print(len(batch_info_list))
