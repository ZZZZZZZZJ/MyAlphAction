"""Train a video classification model."""

import argparse
import os
from tqdm import tqdm
import pickle

import torch
from torch.utils.collect_env import get_pretty_env_info
import torch.distributed as dist
from fvcore.common.timer import Timer

from libs.config import cfg
from libs.dataset import make_data_loader
from libs.modeling.detector import build_detection_model
import libs.modeling.optimizer as optim
from libs.utils.checkpoint import ActionCheckpointer
from libs.dataset.datasets.evaluation import evaluate
from libs.utils.comm import get_rank, is_main_process, all_gather, synchronize, get_world_size
from libs.structures.memory_pool import MemoryPool
from libs.utils.comm import get_rank
from libs.utils.logger import setup_logger
import time

logger = logging.get_logger(__name__)


def train_epoch(train_loader, model, optimizer, cur_epoch, cfg):
    device = torch.device("cuda")
    extra_args = {}
    model.train()
    data_size = len(train_loader)
    iter_timer = Timer()
    mode = 'train'
    
    for iteration, batch in enumerate(train_loader, **extra_args):
        slow_clips, fast_clips, boxes, objects, extras, video_ids = batch
        slow_clips = slow_clips.to(device)
        fast_clips = fast_clips.to(device)
        boxes = [box.to(device) for box in boxes]
        objects = [None if (box is None) else box.to(device) for box in objects]
        
        output = model(slow_clips, fast_clips, boxes, objects, extras)
        output = output[2]
        start = time.time()

        scores = []
        labels = []
        for i in range(len(output)):
            scores.append(output[i].get_field('scores'))
            labels.append(ori_boxes[i].get_field('labels'))
        scores = torch.cat(scores,dim=0).clamp(0,1)
        labels = torch.cat(labels,dim=0).clamp(0,1)
        print(scores.shape, labels.shape)
        
        loss_fun = torch.nn.BCELoss(reduction="mean")
        # Compute the loss.
        loss = loss_fun(scores, labels.type(torch.float))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if cfg.NUM_GPUS > 1:
            loss = du.all_reduce([loss])[0]
        loss = loss.item()
        
        iter_timer.pause()
        eta_sec = iter_timer.seconds() * (data_size - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        
        stats = {
                "_type": "{}_iter".format(mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "time_diff": iter_timer.seconds(),
                "mode": mode,
                "loss": loss,
                "lr": lr,
            }
        logging.log_json_stats(stats)
        iter_timer.reset()

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
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

@torch.no_grad()
def eval_epoch(val_loader, model, cur_epoch, cfg):
    device = torch.device("cuda")
    cpu_device = torch.device("cpu")
    extra_args = {}
    results_dict = {}
    model.eval()
    mode = 'val'
    data_size = len(val_loader)
    iter_timer = Timer()

    for iteration, batch in enumerate(tqdm(eval_loader,**extra_args)):
        slow_clips, fast_clips, boxes, objects, extras, video_ids = batch

        slow_clips = slow_clips.to(device)
        fast_clips = fast_clips.to(device)
        boxes = [box.to(device) for box in boxes]
        ori_boxes = [box.to(device) for box in ori_boxes]
        objects = [None if (box is None) else box.to(device) for box in objects]

        with torch.no_grad():
            output = model(slow_clips, fast_clips, boxes, objects, extras)
            #print(len(output),output[0])
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {video_id: result for video_id, result in zip(video_ids, output)}
        )
        
        iter_timer.pause()
        eta_sec = iter_timer.seconds() * (data_size - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
                "_type": "{}_iter".format(mode),
                "cur_epoch": "{}".format(cur_epoch + 1),
                "cur_iter": "{}".format(cur_iter + 1),
                "eta": eta,
                "time_diff": iter_timer.seconds(),
                "mode": mode,
        }
        logging.log_json_stats(stats)
        iter_timer.reset()  
        
    synchronize()
    predictions = _accumulate_predictions_from_multiple_gpus(results_dict)
    
    if cfg.OUTPUT_DIR:
        torch.save(predictions, os.path.join(cfg.OUTPUT_DIR, "predictions.pth"))

    return evaluate(
        dataset=eval_loader.dataset,
        predictions=predictions,
        output_folder=cfg.OUTPUT_DIR,
    )


def calculate_and_update_precise_bn(loader, model, num_iters=200):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
    """

    def _gen_loader():
        for inputs, _, _, _ in loader:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def train(cfg):
    """
    Train a video model for many epochs on train set and evaluate it on val set.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Train with config:")
    logger.info(pprint.pformat(cfg))

    # Build the video model and print model statistics.
    model = build_detection_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model, cfg, is_train=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Load a checkpoint to resume training if applicable.
    checkpointer = ActionCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR,save_to_disk=True, logger = logger)
    checkpointer.load(cfg.MODEL.WEIGHT)
    state = model.state_dict()
    start_epoch = checkpointer.get_epoch()
    start_epoch = 22
    
    # Create the video train and val loaders.
    data_loader_train = make_data_loader(cfg, is_train=True, is_distributed=distributed)
    data_loader_test = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    # Perform the training loop.
    logger.info("Start epoch: {}".format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset.
        if cfg.TRAIN.ENABLE:
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(cur_epoch)
            # Train for one epoch.
            train_epoch(train_loader, model, optimizer, cur_epoch, cfg)

            # Compute precise BN stats.
            if cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(model)) > 0:
                calculate_and_update_precise_bn(
                    train_loader, model, cfg.BN.NUM_BATCHES_PRECISE
                )
    
            # Save a checkpoint.
            checkpointer.save(os.path.join('checkpoints', 'checkpoint_epoch_' + str(cur_epoch).zfill(5) + '.pyth'), cur_epoch)

        # Evaluate the model on validation set.
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(val_loader, model, cur_epoch, cfg)