# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import os
import json
import random
import time
import pickle
from pathlib import Path
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from engine import evaluate, train_one_epoch
from models import build_model
from datasets.fewshotLoader import roibatchLoader
from pytorchgo.utils import logger
from pytorchgo.utils.pytorch_utils import model_summary, optimizer_summary
from tqdm import tqdm
import pytorchgo_args
import scipy.io
import h5py

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--shot', default=0, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--net', dest='net', default='I3D', type=str, choices=['I3D', 'c3d', 'resnet18', 'resnet34'],
                        help='backbone')
    parser.add_argument('--crop_size', dest='crop_size', default=320, type=int,
                        help='crop size of frames')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=1, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--sup_layers', default=1, type=int,
                        help="Number of support encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=216, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=6, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # action dataset parameters
    parser.add_argument('--dataset', dest='dataset', default='ava', type=str, choices=['ava', 'ucf101-24'],
                        help='training dataset')
    parser.add_argument('--len_train', default=10000, type=int, help="for debug, None: original len")
    parser.add_argument('--len_val', default=500, type=int, help="reducing validation time")
    parser.add_argument('--len_test', default=None, type=int)

    # run name
    parser.add_argument('--run_name', dest='run_name', type=str,
                        help='name of this run')


    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument("--logger_option", default=None, type=str)
    parser.add_argument('--schedulelr', default='cosine', type=str,choices=['cosine','steplr'])


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

def get_roidb(path):
    data = pickle.load(open(path, 'rb'))
    return data
def get_roidb_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def main(args):
    utils.init_distributed_mode(args)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    logger.info(args)
    import getpass
    customized_name = 'train_log/v0_{user}_d{debug}_i3d_bs{bs}shot{shot}epoch{epoch}{schedulelr}lr{lr}lrb{lrb}seed{sd}_enc{enc}sup{sup}dec{dec}head{nheads}q{num}'.\
        format(bs=args.batch_size,
               debug=int(args.debug),
               enc=args.enc_layers,
               dec=args.dec_layers,
               sup=args.sup_layers,
               sd=args.seed,
               lr=args.lr,
               shot=args.shot,
               schedulelr=args.schedulelr,
               epoch=args.epochs,
               nheads=args.nheads,
               num=args.num_queries,
               user=getpass.getuser(),
               lrb=args.lr_backbone)
    # wandb
    args.run_name = customized_name.replace("train_log/", "")

    if args.wandb:
        import wandb
        os.system('wandb login c3be77ba0cba371a692db3aacb4f5ceec8e73440')
        wandb.init(config=args, name=args.run_name, project="few-shot-detr")
        wandb.config.update(dict(name=args.run_name))
    logger.set_logger_dir(customized_name, args.logger_option)
    pytorchgo_args.set_args(args)

    if args.debug:
        logger.warning("debug mode!!!!")
        args.len_train = 60
        args.len_val = 60
        args.len_test = 60
        args.epochs = 3
        args.batch_size = 10
    else:
        pass


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # action dataset
    if args.dataset == "ava":
        args.imdb_name = 'ava_train.pkl'
        args.imdb_val_name = 'ava_val.json'
        args.imdb_test_name = 'ava_test.json'
    else:
        raise

    roidb_path = os.path.join("./datasets/", 'ava_val.json')
    roidb = get_roidb_json(roidb_path)
    print(len(roidb))

    # roidb_path = os.path.join("./datasets/", 'finalAnnots.mat')
    roidb_path = os.path.join("./datasets/", '00025.mat')
    roidb = h5py.File(roidb_path)
    arrays = {}
    for k, v in roidb.items():
        arrays[k] = np.array(v)
    roidb = scipy.io.loadmat(roidb_path)

    roidb_path = os.path.join("./datasets/", 'ucf.pkl')
    roidb = get_roidb(roidb_path)

    for k in roidb.keys():
        if len(roidb[k]['annotations'])>1:
            t = 0

    roidb_path = os.path.join("./datasets/", args.imdb_name)
    roidb = get_roidb(roidb_path)

    # only flipped false
    dataset = roibatchLoader(roidb, crop_size=args.crop_size, len_dataset=args.len_train,
                            shot=args.shot)
    roidb_val_path = "./datasets/" + args.imdb_val_name
    roidb_val = get_roidb_json(roidb_val_path)
    roidb_test_path = "./datasets/" + args.imdb_test_name
    roidb_test = get_roidb_json(roidb_test_path)
    dataset_val = roibatchLoader(roidb_test, crop_size=args.crop_size, len_dataset=args.len_val, phase='test',
                                shot=args.shot)
    dataset_test = roibatchLoader(roidb_test, crop_size=args.crop_size, len_dataset=args.len_test, phase='test',
                                 shot=args.shot)

    if args.distributed:
        sampler_train = DistributedSampler(dataset)
        # sampler_train_4eval = DistributedSampler(dataset_4eval)
        # sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset)
        # sampler_train_4eval = torch.utils.data.RandomSampler(dataset_4eval)
        # sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
    #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)

    data_loader_train = DataLoader(dataset, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # data_loader_train_4eval = DataLoader(dataset_4eval, sampler=sampler_val, drop_last=False,
    #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                  drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    model, criterion, postprocessors = build_model(args)
    device = torch.device(args.device)
    model.to(device)
    logger.info(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    if args.schedulelr == 'steplr':
        lr_drop = args.epochs*2//3
        logger.warning("lr_drop={}".format(lr_drop))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_drop, gamma=0.1, last_epoch=-1)
    elif args.schedulelr == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0, last_epoch=-1)
    else:raise

    optimizer_summary(optimizer)
    model_summary(model)
    start_time = time.time()
    current_time = time.time()
    if args.eval:
        logger.info("Start evaluating")
        checkpoint = torch.load(os.path.join(logger.get_logger_dir(), 'shit.pth'), map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        result = evaluate(model, criterion, data_loader_test, device)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Final result {}'.format(result))
        logger.info('Evaluating time {}'.format(total_time_str))
        return

    logger.info("Start training")
    best_result = 0
    for epoch in tqdm(range(args.start_epoch, args.epochs), desc="training"):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, lr_scheduler,
                        args.batch_size, args.clip_max_norm)
        #lr_scheduler.step()

        epoch_time = time.time() - current_time
        current_time = time.time()
        logger.info('Epoch:{} epoch time:{} '.format(epoch, datetime.timedelta(seconds=int(epoch_time))))
        logger.info("Validation ...")

        result = evaluate(model, criterion, data_loader_test, device)
        # if True:
        #     logger.warning("evaluation on train data..")
        #     train_result = evaluate(model, criterion, data_loader_train_4eval, device)
        #     wandb.log(dict(train_acc=train_result))
        eval_time = time.time() - current_time
        current_time = time.time()
        if result >= best_result:
            best_result = result
            torch.save({
                'model': model_without_ddp.state_dict(),
            }, os.path.join(logger.get_logger_dir(), 'shit.pth'))
            logger.info('best result! shit saved!')

        logger.info('Validation time:{} best_result:{} result:{}'.format(datetime.timedelta(seconds=int(eval_time)),
                                                                   best_result, result))
        wandb.log(
            dict(eval_acc=result, best_acc=best_result))

    current_time = time.time()
    total_time = current_time - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Total training time {}'.format(total_time_str))
    logger.info("Testing ...")
    checkpoint = torch.load(os.path.join(logger.get_logger_dir(), 'shit.pth'), map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    result = evaluate(model, criterion, data_loader_test, device)
    wandb.run.summary["final_test_accuracy"] = result
    test_time = time.time() - current_time
    logger.warning('Congrats! Testing time:{} final result:{}'.format(datetime.timedelta(seconds=int(test_time)), result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
