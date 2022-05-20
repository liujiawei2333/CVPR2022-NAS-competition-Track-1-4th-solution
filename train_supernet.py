from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter
from scipy.stats import kendalltau
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import torch.optim
import argparse
import logging
import random
import torch
import time
import sys
import json
import os

from utils.progress import AverageMeter, ProgressMeter, accuracy
from solver.build import build_optimizer
from solver.lr_scheduler import WarmupCosineLR
from modules.nas_dynamic_model import SuperNet
from modules.sampler import get_width_and_depth_group
from data.DatasetLoader_imagenet import get_dataset_loader_imagenet
from utils.utils import build_args_and_env
from utils.comm import create_exp_dir
from evaluate import nas_eval as nas_eval
import utils.saver as saver
import utils.comm as comm
import loss_ops as loss_ops


parser = argparse.ArgumentParser(description='one-shot weight sharing nas supernet training')
parser.add_argument('--train-config-file', default='./configs/train_supernet_models.yml', type=str,
                    help='training configuration')
parser.add_argument('--common-config-file', default='./configs/common.yml', type=str,
                    help='common configuration')
run_args = parser.parse_args()

local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
if local_rank != -1:
    torch.cuda.set_device(local_rank)


def main():
    args, common_args = build_args_and_env(run_args)
    main_worker(args, common_args)


def main_worker(args, common_args):

    # Create save path
    save_path = './results/' + common_args.save
    try:
        create_exp_dir(save_path, scripts_to_save=None)
    except OSError:
        pass

    try:
        os.mkdir("%s/tensorboard_log/" % save_path)
    except OSError:
        pass

    # Copy the configuration file to log save path
    saver.copy_file(args.config_file, '{}/{}'.format(save_path, os.path.basename(args.config_file)))
    saver.copy_file(common_args.common_config_file,
                    '{}/{}'.format(save_path, os.path.basename(common_args.common_config_file)))

    # Create all parameters
    args.checkpoint_save_path = os.path.join(
        save_path, 'checkpoint.pth.tar'
    )

    # Create log
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    writer = SummaryWriter(os.path.join(save_path, 'tensorboard_log'))

    random.seed(common_args.seed)
    cudnn.benchmark = True
    torch.manual_seed(common_args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(common_args.seed)
    dist.init_process_group("nccl")

    # Build model
    comm.synchronize()
    model = SuperNet(common_args.n_classes,mode="train_supernet",dynamic_resolution=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

    # Build loss
    criterion = loss_ops.CrossEntropyLossSmooth(args.label_smoothing).to(local_rank)
    soft_criterion = loss_ops.KLLossSoft().to(local_rank)

    # Build dataloader
    train_loader, val_loader, train_sampler = get_dataset_loader_imagenet(args,common_args)

    args.n_iters_per_epoch = len(train_loader)

    # Build optimizer and learning rate scheduler
    optimizer = build_optimizer(args, model)
    if args.lr_scheduler.lr_way == "warmup_cosine_lr":
        args.max_iters = args.n_iters_per_epoch * args.epochs
        args.warmup_iters = args.n_iters_per_epoch * args.warmup_epochs
        warmup_lr = float(getattr(args.lr_scheduler, 'warmup_lr', 0.001))
        warmup_method = getattr(args.lr_scheduler, 'warmup_method', 'linear')
        clamp_lr_percent = float(getattr(args.lr_scheduler, 'clamp_lr_percent', 0.))
        clamp_lr = args.lr_scheduler.base_lr * clamp_lr_percent

        lr_scheduler = WarmupCosineLR(
                optimizer, 
                args.max_iters, 
                warmup_factor = warmup_lr,
                warmup_iters = args.warmup_iters,
                warmup_method = warmup_method,
                clamp_lr = clamp_lr,
        )

    # In case of interruption, load parameters and continue training
    if args.resume:
        saver.load_checkpoints(args, model, optimizer, lr_scheduler, logging)

    subnets_to_be_evaluated = {
        'nas_min_net': {},
        'nas_max_net': {},
    }

    json_file = './json_file/gt_arch30.json'#ACC of stand alone trained 30 subnets

    with open(json_file) as json_file:
        json_file = json.load(json_file)

    gt_kt_list = []
    # The subnets waiting to be evaluated include the max-subnets, the min-subnets, and 30 stand alone trained subnets
    for json_key in json_file.keys():
        subnets_to_be_evaluated[json_file[json_key]["arch"]] = {}
        gt_kt_list.append(json_file[json_key]["acc"])

    logging.info(args)

    for epoch in range(args.start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)

        acc1, acc5, data_for_bn_calibration = train_epoch(epoch, model, train_loader, optimizer, criterion,
            writer, common_args, args, soft_criterion=soft_criterion, lr_scheduler=lr_scheduler)

        pre_kt_list = validate(epoch, subnets_to_be_evaluated, data_for_bn_calibration,
                        val_loader, model, criterion, writer, args, common_args)

        Kendallta , _ = kendalltau(gt_kt_list,pre_kt_list)
        if dist.get_rank() == 0:
            logging.info('Kendallta {}'.format(Kendallta))
        writer.add_scalar('val/Kendallta', Kendallta, epoch)

        # In case of interruption, save parameters and continue training
        if dist.get_rank()  == 0:
            saver.save_checkpoint(
                args.checkpoint_save_path,
                model,
                optimizer,
                lr_scheduler,
                args,
                epoch,
            )

        # Save candidate models
        if dist.get_rank()  == 0:
            if Kendallta > 0.64:
                torch.save(model.state_dict(), os.path.join(save_path, "supernet_%s_%d.pth" % (str(Kendallta),epoch)))


def train_epoch(epoch, model, train_loader, optimizer, criterion, writer, common_args, args, soft_criterion=None, lr_scheduler=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    writer.add_scalar('learning_rate/lr', optimizer.param_groups[0]['lr'], epoch)
    if local_rank == 0:
        logging.info('Training lr {}'.format(optimizer.param_groups[0]['lr']))

    model.train()
    end = time.time()

    num_updates = epoch * len(train_loader)
    data_for_bn_calibration = []

    for batch_idx, (images, target) in enumerate(train_loader):

        # Save a small part of the data to prepare for the recalculation of BN layer parameters in the validate stage
        if batch_idx < common_args.post_bn_calibration_batch_num:
            data_for_bn_calibration.append(images)

        data_time.update(time.time() - end)

        images = images.to(local_rank, non_blocking=True)
        target = target.to(local_rank, non_blocking=True)

        num_subnet_training = max(2, getattr(args, 'num_arch_training', 2))
        optimizer.zero_grad()

        ### Sandwich rule ###
        # stage1：The max-subnet is sampled and only the max-subnet is regularized
        drop_connect_only_last_two_stages = getattr(args, 'drop_connect_only_last_two_stages', True)
        model.module.sample_max_subnet()
        model.module.set_dropout_rate(args.dropout, args.drop_connect, drop_connect_only_last_two_stages)
        output = model(images)
        loss = criterion(output, target)
        loss.backward()

        with torch.no_grad():
            soft_logits = output.clone().detach()

        # stage2：Sample out the min-subnet and several subnets of random size
        sandwich_rule = getattr(args, 'sandwich_rule', True)
        model.module.set_dropout_rate(0, 0, drop_connect_only_last_two_stages)
        for arch_id in range(1, num_subnet_training):
            if arch_id == num_subnet_training - 1 and sandwich_rule:
                model.module.sample_min_subnet()
                output = model(images)
                loss = soft_criterion(output, soft_logits)
                loss.backward()
            else:
                # Random sampling
                if args.sample_way == "random":
                    model.module.sample_active_subnet()
                    output = model(images)
                    loss = soft_criterion(output, soft_logits)
                    loss.backward()
                # Fair sampling from FairNAS:https://arxiv.org/abs/1907.01845.pdf
                elif args.sample_way == "fair":
                    fair_args = {
                        "stem_basic_channel": model.module.stem_basic_channel,
                        "stage_basic_channels": model.module.stage_basic_channels,
                        "expand_ratio_list": model.module.expand_ratio_list,
                        "block_nums": model.module.block_nums,
                        "group_nums": len(model.module.expand_ratio_list),
                        "fair_width": True,
                        "fair_depth": True
                    }

                    width_group, depth_group = get_width_and_depth_group(**fair_args)

                    for width, depth in zip(width_group, depth_group):

                        sample_cfg_for_list = lambda candidates, sample_min, sample_max: \
                        candidates[0] if sample_min else (candidates[-1] if sample_max else random.choice(candidates))

                        resolution_list = [[128,128], [160,160], [192,192],[224,224]]
                        resolution = sample_cfg_for_list(resolution_list, False, False)

                        model.module.set_active_subnet(resolution, width, depth)
                        output = model(images)
                        loss = soft_criterion(output, soft_logits)
                        loss.backward()

        # Clip gradients if specfied
        if getattr(args, 'grad_clip_value', None):
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)

        optimizer.step()

        # Accuracy measured on the local batch
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        corr1, corr5, loss = acc1*args.batch_size, acc5*args.batch_size, loss.item()*args.batch_size #just in case the batch size is different on different nodes
        stats = torch.tensor([corr1, corr5, loss, args.batch_size], device=local_rank)
        dist.barrier()  # synchronizes all processes
        dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM) 
        corr1, corr5, loss, batch_size = stats.tolist()
        acc1, acc5, loss = corr1/batch_size, corr5/batch_size, loss/batch_size
        losses.update(float(loss), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        num_updates += 1

        # Declining learning rate
        if args.lr_scheduler.lr_way == "warmup_cosine_lr":
            if lr_scheduler is not None:
                lr_scheduler.step()

        # Print information about the training phase
        if batch_idx % (args.n_iters_per_epoch // 4) == 0:
            if local_rank == 0:
                progress.display(batch_idx, logging)

        writer.add_scalar('train/loss', losses.get_avg(), epoch)
        writer.add_scalar('train/top1', top1.get_avg(), epoch)
        writer.add_scalar('train/top5', top5.get_avg(), epoch)

    return top1.avg, top5.avg, data_for_bn_calibration


def validate(epoch, subnets_to_be_evaluated, data_for_bn_calibration, val_loader, model, criterion, 
                writer, args, common_args):

    pre_kt_list = nas_eval.validate(epoch, subnets_to_be_evaluated, data_for_bn_calibration, 
        val_loader, model, criterion, writer, args, common_args, local_rank, logging, bn_calibration = True)

    return pre_kt_list


if __name__ == '__main__':
    main()