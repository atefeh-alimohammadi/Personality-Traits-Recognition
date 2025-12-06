#!/usr/bin/env python
# coding: utf-8

# In[4]:


import argparse
import sys
import os
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from functools import partial
from pathlib import Path
from collections import OrderedDict
import random
from utils.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma
from utils.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
import torch.multiprocessing as mp
import pickle
from datasets.datasets import FrameDataset 
from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_samples_collate
import utils
import utils.loss
import model_finetuning
from torch.utils.data import DataLoader
from SDL import *
from landmark.backbone import Landmark
import pandas as pd
import matplotlib.pyplot as plt
import csv

def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video regression', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)  ##def:64 
    parser.add_argument('--epochs', default=50, type=int)     ##def:30
    parser.add_argument('--update_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)

    # Model parameters
    # parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
    #                     help='Name of model to train')
    parser.add_argument('--model', default='s2d_base_patch16_224', type=str, metavar='MODEL',
                    help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--input_size', default=112, type=int,
                        help='videos input size')

    ##added
    parser.add_argument('--scale_factor', type=float, default=1.0, help='Scaling factor for adapter')
    parser.add_argument('--dropout_ratio', type=float, default=0.1, help='Dropout ratio for the model head') #def=0.1
    parser.add_argument('--K', type=int, default=2, help='Default value for K is 2')
    parser.add_argument('--qs', type=int, default=16, help='Default value for qs is 16')
    parser.add_argument('--fix_lgp', action='store_true',
                        default=False, help='Fix landmark guided prompter')


    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)
    parser.add_argument('--model_ema', action='store_true', default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay (default: 0.05)') #def=0.05
    parser.add_argument('--weight_decay_end', type=float, default=0.3, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""") #def=none

    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)') #def=1e-5
    parser.add_argument('--layer_decay', type=float, default=1.0) #def=1

    parser.add_argument('--warmup_lr', type=float, default=1e-7, metavar='LR',
                        help='warmup learning rate (default: 1e-7)') #def=1e-7
    parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)') #def-1e-8

    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR, if scheduler supports') #def=2
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')

    # # Augmentation parameters (Not typically used in regression but kept for consistency)
    # parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
    #                     help='Color jitter factor (default: 0.4)')
    # parser.add_argument('--num_sample', type=int, default=1,
    #                     help='Repeated_aug (default: 1)')
    # parser.add_argument('--aa', type=str, default='rand-m7-n4-mstd0.5-inc1', metavar='NAME',
    #                     help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)'),
    # parser.add_argument('--smoothing', type=float, default=0.0,
    #                     help='Label smoothing (set to 0.0 for regression)')
    # parser.add_argument('--train_interpolation', type=str, default='bicubic',
    #                     help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    # parser.add_argument('--augment', action='store_true', default=False, help='Use data augment')
    # parser.set_defaults(augment=False)
    
    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=1) ## def:5
    parser.add_argument('--test_num_crop', type=int, default=1) ## def:3
    
    # Random Erase params (not typically used in regression)
    parser.add_argument('--reprob', type=float, default=0.0, metavar='PCT',
                        help='Random erase prob (set to 0.0 for regression)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # Mixup params (Typically not used in regression)
    # parser.add_argument('--mixup', type=float, default=0.0,
    #                     help='mixup alpha, set to 0.0 for regression.')
    # parser.add_argument('--cutmix', type=float, default=0.0,
    #                     help='cutmix alpha, set to 0.0 for regression.')
    # parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
    #                     help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    # parser.add_argument('--mixup_prob', type=float, default=0.0,
    #                     help='Probability of performing mixup or cutmix when either/both is enabled')
    # parser.add_argument('--mixup_switch_prob', type=float, default=0.0,
    #                     help='Probability of switching to cutmix when both mixup and cutmix enabled')
    # parser.add_argument('--mixup_mode', type=str, default='batch',
    #                     help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    # parser.add_argument('--ban_mixup', action='store_true', default=True, help='ban mixup')
    # parser.set_defaults(ban_mixup=True)
    
    # Finetuning params
    parser.add_argument('--finetune', default='E:/facial_lanmarking(dataset)/pretrain-AffectNet-7.pth', help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/your_dataset', type=str,
                        help='dataset path')
    parser.add_argument('--train_label_path', default='/path/to/your_dataset/train.csv', type=str,
                        help='train label path')
    parser.add_argument('--test_label_path', default='/path/to/your_dataset/test.csv', type=str,
                        help='test label path')
    parser.add_argument('--eval_data_path', default=None, type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=5, type=int,
                        help='number of the regression targets (e.g., 5 personality traits)')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_segments', type=int, default=1)
    parser.add_argument('--num_frames', type=int, default=6)
    parser.add_argument('--sampling_rate', type=int, default=1)
    parser.add_argument('--data_set', default='ChaLearnFirstImpressions', choices=['ChaLearnFirstImpressions'],
                        type=str, help='dataset')
    parser.add_argument('--output_dir', default='F:\\DeepPersonality\\ChaLearn2017\\outpus',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='F:\\DeepPersonality\\ChaLearn2017\\outpus',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument('--no_save_ckpt', action='store_false', dest='save_ckpt')
    parser.set_defaults(save_ckpt=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--sdl', type=bool, default=False, help='Description of what this argument does')
    parser.add_argument('--enable_deepspeed', action='store_true', default=False)

    # Regression-specific parameters
    parser.add_argument('--val_metric', type=str, default='loss', choices=['loss', 'mae', 'mse', 'rmse', 'r2'],
                        help='validation metric for saving best ckpt')

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig
            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init

def save_losses(training_losses, validation_loss, file_path="losses.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump({"training_losses": training_losses, "validation_loss": validation_loss}, f)

# تابع بارگذاری مقادیر
def load_losses(file_path="losses.pkl"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data["training_losses"], data["validation_loss"]
    else:
        return [], [] 



def plot_and_save_individual_curves(training_loss, validation_loss, output_dir):
    # اطمینان از وجود پوشه خروجی
    os.makedirs(output_dir, exist_ok=True)
    
    # رسم نمودار Training Loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(training_loss) + 1), training_loss, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.grid()
    training_loss_path = os.path.join(output_dir, 'training_loss_curve.png')
    # print(f"Saving plot1 to: {path}")
    plt.savefig(training_loss_path)
    plt.close()

    # رسم نمودار Validation Loss (اگر داده وجود داشته باشد)
    if validation_loss:
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(validation_loss) + 1), validation_loss, label='Validation Loss', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Accuracy')
        plt.title('Validation Mean Accuracy over Epochs')
        plt.legend()
        plt.grid()
        validation_loss_path = os.path.join(output_dir, 'validation_loss_curve.png')
        # print(f"Saving plot2 to: {path}")
        plt.savefig(validation_loss_path)
        plt.close()


def main(args, ds_init):   # حذف local_rank و nprocs
    # args.local_rank = local_rank
    # args.world_size = nprocs
    # utils.init_distributed_mode(args)
    training_losses, validation_loss = load_losses(file_path=os.path.join(args.output_dir, "losses.pkl"))
    print("init !")
    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed  # حذف utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    ##dataset_train, args.nb_classes = build_dataset(is_train=True, test_mode=False, args=args)
    train_dataset = FrameDataset(root_dir='F:\\DeepPersonality\\ChaLearn2017\\train_face_landmark_img_dir' , labels_pickle='F:\DeepPersonality\\annotation_training.pkl', input_size=112)
    validation_dataset = FrameDataset(root_dir='F:\\DeepPersonality\\ChaLearn2017\\valid_face_landmark_img_dir', labels_pickle='F:\\DeepPersonality\\annotation_validation.pkl', input_size=112)
    test_dataset = FrameDataset(root_dir='F:\\DeepPersonality\\ChaLearn2017\\test', labels_pickle='F:\\DeepPersonality\\annotation_test.pkl', input_size=112)
   ## if args.disable_eval_during_finetuning:
       ## dataset_val = None
   ## else:
        ##dataset_val, _ = build_dataset(is_train=False, test_mode=False, args=args)
        ##validation_dataset = FrameDataset(root_dir='dataset/validation', labels_pickle='path/to/labels.pkl', input_size=224)

    ##dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args)
    ##test_dataset = FrameDataset(root_dir='dataset/test', labels_pickle='path/to/labels.pkl', input_size=224)


    # num_tasks = utils.get_world_size()
    # global_rank = utils.get_rank()

    # sampler_train = None  # نیازی به نمونه‌گیر خاصی نیست، DataLoader از shuffle استفاده می‌کند
    # print("Sampler_train = %s" % str(sampler_train))
    # if args.dist_eval:
    #     sampler_val = torch.utils.data.DistributedSampler(
    #         dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    #     sampler_test = torch.utils.data.DistributedSampler(
    #         dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    # else:
    #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    #     sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # if global_rank == 0 and args.log_dir is not None:
    #     os.makedirs(args.log_dir, exist_ok=True)
    #     log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    # else:
    #     log_writer = None

    # collate_func = None  # For regression, we typically don't use custom collate functions

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    # collate_func = None
    
    
    # validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    #     collate_fn=collate_func,
    # )
    data_loader_train = DataLoader(train_dataset,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=args.pin_mem,
                                   drop_last=True,)

    print("After initializing DataLoader:")
    print(torch.cuda.memory_summary())
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())

    # if dataset_val is not None:
    #     data_loader_val = torch.utils.data.DataLoader(
    #         dataset_val, sampler=sampler_val,
    #         batch_size=args.batch_size,  # Adjust batch size if needed
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=False
    #     )
    # else:
    #     data_loader_val = None
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=32,
                                   shuffle=True,
                                   num_workers=4,
                                   pin_memory=args.pin_mem,
                                   drop_last=False,)
    

    # if dataset_test is not None:
    #     data_loader_test = torch.utils.data.DataLoader(
    #         dataset_test, sampler=sampler_test,
    #         batch_size=args.batch_size,
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_mem,
    #         drop_last=False
    #     )
    # else:
    #     data_loader_test = None
    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=args.pin_mem,
                             drop_last=False,)
    

    # mixup_fn = None  # Mixup is typically not used in regression tasks

    # if args.model.startswith('vit') or args.model.startswith('ViT'):
    if args.model == 's2d_base_patch16_224':
    
        # in_chans_l = 128 if not args.backbone.startswith('MobileNet') else 96
        # if args.backbone == 'MobileNetV2_56':
        #     in_chans_l = 24
        in_chans_l = 3 #prev :1 
        model = create_model(
            args.model,
            pretrained=args.finetune,
            num_classes=args.nb_classes,  # Number of regression targets
            adapter_scale=args.scale_factor,
            head_dropout_ratio=args.dropout_ratio,
            num_frames=args.num_frames * args.num_segments,
            in_chans_l=in_chans_l
        )
        # Replace the classification head with a regression head
        hidden_dim = model.head.in_features
        model.head = torch.nn.Linear(hidden_dim, args.nb_classes)
    else:
        raise NotImplementedError("Model architecture not supported.")
        
    sdl=None
    if args.sdl:
        sdl = SDL(args.nb_classes, k=args.K, size=args.qs).to(device)
    else:
        print(f'SDL: {args.sdl}, do not use sdl loss')
        
    # Initialize landmark model if needed
    ##model2 = Landmark(args.backbone)
    # model2 = Landmark()
    
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    # print(f'Modified {args.model} head for regression: {model.get_classifier()}')
    
    # Freeze certain parameters if needed
    # for name, param in model.named_parameters():
    #     if 'head' in name or 'cls_token' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    for name, param in model.named_parameters():
        if 'temporal_embedding' not in name and 'temporal_attn' not in name and 'cls_token_t' not in name and 'ln_post' not in name and 'cls_head' not in name and 'Adapter' not in name and 'head' not in name and 'prompt' not in name:
            param.requires_grad = False
        if args.fix_lgp and 'prompt' in name and 'Adapter' not in name: 
            param.requires_grad = False
            

    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])##args.num_frames//2 ??
    args.patch_size = patch_size

    model.to(device)

    print("After moving model to GPU:")
    print(torch.cuda.memory_summary())
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())
    
    # model2.to(device)
    # model2.eval()

    model_ema = None ##؟؟؟؟
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    # model_without_ddp = model ##؟؟؟؟
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print model parameter info
    print('number of params:', n_parameters)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    num_total_param = sum(p.numel() for p in model.parameters()) / 1e6
    print('Number of total parameters: {} M, tunable parameters: {} M'.format(num_total_param, num_param))
    
    total_batch_size = args.batch_size * args.update_freq ##؟؟؟؟؟
    num_training_steps_per_epoch = len(train_dataset) // total_batch_size
    
    ## settings of learning rate
    args.lr = args.lr * total_batch_size / 8
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(train_dataset))
    print("Number of training steps per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0:
        num_layers = model.get_num_layers()
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None
        
    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    # if args.enable_deepspeed:
    #     loss_scaler = None
    #     optimizer_params = get_parameter_groups(
    #         model, args.weight_decay, skip_weight_decay_list,
    #         assigner.get_layer_id if assigner is not None else None,
    #         assigner.get_scale if assigner is not None else None)
    #     model, optimizer, _, _ = ds_init(
    #         args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
    #     )

    #     print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
    #     assert model.gradient_accumulation_steps() == args.update_freq
    # else:
    #     if args.distributed:
    #         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #         model_without_ddp = model.module

    #     optimizer = create_optimizer(
    #         args, model_without_ddp, skip_list=skip_weight_decay_list,
    #         get_num_layer=assigner.get_layer_id if assigner is not None else None, 
    #         get_layer_scale=assigner.get_scale if assigner is not None else None)
    #     loss_scaler = NativeScaler()
    
    optimizer = create_optimizer(
    args, model, skip_list=skip_weight_decay_list,
    get_num_layer=assigner.get_layer_id if assigner is not None else None, 
    get_layer_scale=assigner.get_scale if assigner is not None else None)

    # تنظیم loss scaler
    loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    print(f"Learning rate: {args.lr}")
    print(f"Minimum learning rate: {args.min_lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Steps per epoch: {num_training_steps_per_epoch}")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    
    # Use MSELoss for regression
    criterion = torch.nn.MSELoss()
    sdl_loss_fn=None
    if args.sdl:
        sdl_loss_fn = utils.loss.BCEWithLogitsLoss()
        print("sdl_loss_fn = %s" % str(sdl_loss_fn))
        
    criterion = { 
        'criterion': criterion,
        'sdl_loss_fn': sdl_loss_fn
     }    
    print("criterion = %s" % str(criterion))
    
    #evaluation   
    utils.auto_load_model(
        args=args, model=model,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

  
            
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_metric = float('inf') if args.val_metric == 'loss' else -float('inf')
    best_epoch = None
    
    mixup_fn = None
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch} - Before training step:")
        print(torch.cuda.memory_summary())
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model, sdl, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq
        )
        training_losses.append(train_stats['loss'])

        print(f"Epoch {epoch} - After training step:")
        print(torch.cuda.memory_summary())
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch} completed in {str(datetime.timedelta(seconds=int(epoch_time)))}")
    
        
        if args.output_dir and args.save_ckpt:
            utils.save_model(
                args=args, model=model, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        if validation_loader is not None:
            print("Before validation:")
            print(torch.cuda.memory_summary())
            print(torch.cuda.memory_allocated())
            print(torch.cuda.memory_reserved())
            preds_file = os.path.join(args.output_dir, f"val_predictions_epoch_{epoch}.txt")
            val_stats = validation_one_epoch(validation_loader, model, args, device, regression=True)
            validation_loss.append(val_stats['metrics']['ma'])
            print("After validation:")
            print(torch.cuda.memory_summary())
            print(torch.cuda.memory_allocated())
            print(torch.cuda.memory_reserved())
            preds, labels = val_stats['outputs'], val_stats['targets']  # فرض: validation_one_epoch خروجی‌ها را بازمی‌گرداند
            mse = np.mean((preds - labels) ** 2)
            mae = np.mean(np.abs(preds - labels))
            rmse = np.sqrt(mse)
            r2 = 1 - (np.sum((labels - preds) ** 2) / np.sum((labels - np.mean(labels)) ** 2))
            print(f"Validation Metrics for Epoch {epoch}:")
            print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            print(f"Validation Metrics: {val_stats}")
                    # رسم نمودار Scatter Plot
            plt.figure(figsize=(8, 8))
            plt.scatter(labels, preds, alpha=0.5)
            plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], color='red', linestyle='--', linewidth=2)
            plt.xlabel("True Values")
            plt.ylabel("Predicted Values")
            plt.title(f"Validation: True vs Predicted Values (Epoch {epoch})")
            plt.grid()
            plt.savefig(os.path.join(args.output_dir, f"scatter_plot_val_epoch_{epoch}.png"))
            plt.show()
        
            # رسم نمودار Residual Plot
            residuals = labels - preds
            plt.figure(figsize=(8, 6))
            plt.scatter(preds, residuals, alpha=0.5)
            plt.axhline(0, color='red', linestyle='--', linewidth=2)
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title(f"Validation: Residual Plot (Epoch {epoch})")
            plt.grid()
            plt.savefig(os.path.join(args.output_dir, f"residual_plot_val_epoch_{epoch}.png"))
            plt.show()
            
            # # Save best model based on selected metric
            # if (args.val_metric == 'loss' and val_stats['metrics']['mse'] < best_metric):
            #     best_metric = val_stats[args.]
            #     best_epoch = epoch
            #     if args.output_dir and args.save_ckpt:
            #         utils.save_model(
            #             args=args, model=model, optimizer=optimizer,
            #             loss_scaler=loss_scaler, epoch=epoch, is_best=True)
            
            # print(f"Best '{args.val_metric.upper()}': {best_metric} (epoch {best_epoch})")
            if log_writer is not None:
                for key, value in val_stats.items():
                    log_writer.update({f'val_{key}': value}, step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            torch.cuda.empty_cache()
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
        save_losses(training_losses, validation_loss, file_path=os.path.join(args.output_dir, "losses.pkl"))
        if args.output_dir:
            if log_writer is not None:
                log_writer.flush()
                
            for key, value in log_stats.items():
                if isinstance(value, np.ndarray):  # اگر مقدار از نوع ndarray است
                    log_stats[key] = value.tolist()  
                    
            with open(os.path.join(args.output_dir, "log.txt"), mode="w", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
                f.write(f"Epoch {epoch} Validation Metrics:\n")
                f.write(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}\n")
        remain_time = (time.time() - start_time)/(epoch - args.start_epoch + 1) * (args.epochs - epoch - 1)
        remain_time_str = str(datetime.timedelta(seconds=int(remain_time)))
        print(f"Remaining time: {remain_time_str}")

    # Final evaluation on test set using best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoint-best.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    preds_file = os.path.join(args.output_dir, 'predictions.txt')
    torch.cuda.empty_cache()
    ##تنطیم تست هم دیتا هم لند مارک
    test_stats = final_test(test_loader, model, args, device, preds_file, regression=True)

    # استخراج خروجی‌ها و مقادیر واقعی
    preds, labels = test_stats['outputs'], test_stats['targets']  
    print(preds[:5]) 
    fixed_preds = []
    for p in preds:
        if len(p) < 32:
            p = np.pad(p, (0, 32 - len(p)), mode='constant', constant_values=0)
        fixed_preds.append(p)
 

    # for i, arr in enumerate(fixed_preds):
    #     print(f"Element {i}: shape={np.shape(arr)}")
    fixed_preds_fixed = []
    expected_shape = (32, 5)  # شکل مورد انتظار

    for i, arr in enumerate(fixed_preds):
        arr_np = np.array(arr)  # تبدیل به NumPy
        if arr_np.shape != expected_shape:  # اگر شکل درست نیست
            print(f"Fixing element {i}: shape={arr_np.shape}")
            
            # اگر تعداد ستون‌ها بیشتر است، برش دهید
            if arr_np.shape[1] > expected_shape[1]:
                arr_np = arr_np[:, :expected_shape[1]]
            
            # اگر تعداد ستون‌ها کمتر است، با صفر پر کنید
            elif arr_np.shape[1] < expected_shape[1]:
                padding = expected_shape[1] - arr_np.shape[1]
                arr_np = np.pad(arr_np, ((0, 0), (0, padding)), mode='constant')
        
        fixed_preds_fixed.append(arr_np)
    # تبدیل لیست اصلاح‌شده به NumPy
    fixed_preds_np = np.array(fixed_preds_fixed)
    print(fixed_preds_np.shape)  # باید (368, 32, 5) باشد    
    preds = fixed_preds_np
    labels_fixed = []
    expected_shape = (32, 5)  # شکل مورد انتظار
    
    for i, label in enumerate(labels):
        label_np = np.array(label)  # تبدیل به NumPy
        if label_np.shape != expected_shape:  # اگر شکل درست نیست
            print(f"Fixing label {i}: shape={label_np.shape}")
            
            # اگر تعداد ستون‌ها بیشتر است، برش دهید
            if label_np.shape[1] > expected_shape[1]:
                label_np = label_np[:, :expected_shape[1]]
            
            # اگر تعداد ستون‌ها کمتر است، با صفر پر کنید
            elif label_np.shape[1] < expected_shape[1]:
                padding = expected_shape[1] - label_np.shape[1]
                label_np = np.pad(label_np, ((0, 0), (0, padding)), mode='constant')
            # اگر تعداد ردیف‌ها کمتر است، با صفر پر کنید
            if label_np.shape[0] < expected_shape[0]:
                padding_rows = expected_shape[0] - label_np.shape[0]
                label_np = np.pad(label_np, ((0, padding_rows), (0, 0)), mode='constant')    
        
        labels_fixed.append(label_np)  # اضافه کردن آرایه اصلاح‌شده
    for i, arr in enumerate(labels_fixed):
        print(f"Element {i}: shape={np.shape(arr)}")
    labels = np.array(labels_fixed)
    print(labels.shape)
    mse = np.mean((preds - labels) ** 2)
    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((labels - preds) ** 2) / np.sum((labels - np.mean(labels)) ** 2))
    
    print("Final Test Metrics:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # رسم نمودار Scatter Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(labels, preds, alpha=0.5)
    plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], color='red', linestyle='--', linewidth=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Test: True vs Predicted Values")
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, "scatter_plot_test.png"))
    plt.show()
    
    # رسم نمودار Residual Plot
    residuals = labels - preds
    plt.figure(figsize=(8, 6))
    plt.scatter(preds, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Test: Residual Plot")
    plt.grid()
    plt.savefig(os.path.join(args.output_dir, "residual_plot_test.png"))
    plt.show()

    ## plot for training loss and validtation mse over epochs
    
    plot_and_save_individual_curves(training_losses, validation_loss, args.output_dir)

  



if __name__ == '__main__':
    opts, ds_init = get_args()
    
    print(opts.train_label_path, opts.test_label_path, opts.save_ckpt_freq)
    
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    
    # تعیین دستگاه (GPU یا CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # اجرای مستقیم تابع main به‌صورت تک‌پردازشی
    main(args=opts, ds_init=ds_init)
    print("At the end of the program:")
    print(torch.cuda.memory_summary())
    print(torch.cuda.memory_allocated())
    print(torch.cuda.memory_reserved())






