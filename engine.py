# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import torch.nn.functional as F
import torch
import numpy as np
import csv

import util.misc as utils
from util.eval_detection import ANETdetection
import wandb
from tqdm import tqdm
from pytorchgo.utils import logger
from get_ava_performance import run_evaluation
import pytorchgo_args

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, lr_scheduler,batch_size: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    logger.warning("training epoch={}".format(epoch))
    for iter_num, (samples, supports, targets, _) in tqdm(enumerate(data_loader), total=len(data_loader),desc='train epoch={}/{}'.format(epoch,pytorchgo_args.get_args().epochs)):
        if True:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            shot = model.shot
            if shot > 0:
                supports = supports.to(device)  # torch.Size([8, 3, 3, 64, 112, 112]) batch, shot, chanel, TWH

            outputs = model(samples, supports)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                          for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            lr_scheduler.step(iter_num*1.0/len(data_loader)+epoch)

            if (iter_num*batch_size) % 600 == 0:
                wandb.log(dict(loss=loss_value,class_error=loss_dict_reduced['class_error'],lr=optimizer.state_dict()['param_groups'][0]["lr"],
                               loss_ce_scaled=loss_dict_reduced_scaled['loss_ce'],
                    loss_bbox_scaled=loss_dict_reduced_scaled['loss_bbox'],
                    loss_giou_scaled=loss_dict_reduced_scaled['loss_giou'],
                    loss_ce=loss_dict_reduced['loss_ce'],
                    loss_bbox=loss_dict_reduced['loss_bbox'],
                    loss_giou=loss_dict_reduced['loss_giou']))
                logger.info("loss={loss}, lr={lr}".format(loss=round(loss_value,4),lr=round(optimizer.state_dict()['param_groups'][0]["lr"],8)))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    #print("Averaged stats:", metric_logger)
    return


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    prediction = []
    groundtruth = []

    for iter_num, (samples, supports, targets, video_ids) in tqdm(enumerate(data_loader), total=len(data_loader),desc='eval'):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        shot = model.shot
        if shot > 0:
            supports = supports.to(device)

        outputs = model(samples, supports)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        # prediction
        pred_cls = F.softmax(outputs['pred_logits'], dim=2)
        pred_box = outputs['pred_boxes']
        bs, num_query, num_cls = pred_cls.shape
        for i in range(bs):
            for j in range(num_query):
                for k in range(1, num_cls):
                    x1 = max(0, pred_box[i][j][0].item() - pred_box[i][j][2].item() / 2)
                    y1 = max(0, pred_box[i][j][1].item() - pred_box[i][j][3].item() / 2)
                    x2 = min(1, pred_box[i][j][0].item() + pred_box[i][j][2].item() / 2)
                    y2 = min(1, pred_box[i][j][1].item() + pred_box[i][j][3].item() / 2)
                    prediction.append([video_ids[i], 902, x1, y1, x2, y2, k, pred_cls[i][j][k].item()])
        # ground truth
        for i in range(bs):
            x1 = max(0, targets[i]['boxes'].cpu()[0][0].item() - targets[i]['boxes'].cpu()[0][2].item() / 2)
            y1 = max(0, targets[i]['boxes'].cpu()[0][1].item() - targets[i]['boxes'].cpu()[0][3].item() / 2)
            x2 = min(1, targets[i]['boxes'].cpu()[0][0].item() + targets[i]['boxes'].cpu()[0][2].item() / 2)
            y2 = min(1, targets[i]['boxes'].cpu()[0][1].item() + targets[i]['boxes'].cpu()[0][3].item() / 2)
            groundtruth.append([video_ids[i], 902, x1, y1, x2, y2, 1, 1])
    # evaluation
    result = run_evaluation(groundtruth, prediction)  # mAP0.5

    metric_logger.synchronize_between_processes()
    return result
