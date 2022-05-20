from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data.distributed
import torch.distributed as dist
import torch.nn.parallel
import torch.utils.data
import torch.optim
import argparse
import logging
import random
import torch
import sys
import json
import os

from modules.nas_dynamic_model import SuperNet
from data.DatasetLoader_imagenet import get_dataset_loader_stand_alone
from utils.progress import AverageMeter, ProgressMeter, accuracy
from solver import build_optimizer
from utils.flops_counter import count_net_flops_and_params
from utils.utils import build_args_and_env, code2list
from solver.lr_scheduler import WarmupCosineLR


parser = argparse.ArgumentParser(description='stand alone subnet training')
parser.add_argument('--train-config-file', default='./configs/train_supernet_models.yml', type=str,
                    help='training configuration')
parser.add_argument('--common-config-file', default='./configs/common.yml', type=str,
                    help='common configuration')
parser.add_argument('--target_arch_num', default=1, type=int)
run_args = parser.parse_args()

local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
if local_rank != -1:
    torch.cuda.set_device(local_rank)

# Create log
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')


def main():
    args, common_args = build_args_and_env(run_args)
    main_worker(args, common_args)


def main_worker(args, common_args):

    random.seed(common_args.seed)
    cudnn.benchmark = True
    torch.manual_seed(common_args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(common_args.seed)
    dist.init_process_group("nccl")

    # Parameter
    target_arch_num = run_args.target_arch_num
    stem_base_channel = 64
    block_base_channels = [64, 128, 256, 512]
    expand_ratio_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

    # Build model
    model = SuperNet(common_args.n_classes,"train_stand_alone")
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

    json_file = './json_file/gt_arch30.json'#ACC of stand alone trained 30 subnets

    with open(json_file) as json_file:
        json_file = json.load(json_file)

    # Find the code of the target subnet
    for key_index, json_key in enumerate(json_file.keys()):
        if key_index == target_arch_num - 1:
            arch = json_file[json_key]["arch"]
            break

    logging.info(arch)

    # Sample the target subnet
    width, depth = code2list(arch, stem_base_channel, block_base_channels, expand_ratio_list)
    resolution = [224,224]
    model.module.set_active_subnet(resolution, width, depth)
    subnet = model.module.get_active_subnet()

    # The flops and params of the target subnet
    data_shape = (1, 3, 224, 224)
    flops, params = count_net_flops_and_params(subnet, data_shape)
    logging.info("flops:{} M".format(flops))
    logging.info("params:{} M".format(params))

    del subnet

    args.batch_size = 256
    args.epochs = 90
    args.warmup_epochs = 2

    train_loader, val_loader, train_sampler = get_dataset_loader_stand_alone(args,common_args)

    args.n_iters_per_epoch = len(train_loader)
    args.weight_decay_weight = 1e-4
    args.lr_scheduler.base_lr = 0.2

    # Build loss
    criterion = nn.CrossEntropyLoss().to(local_rank)

    # Build optimizer and learning rate scheduler
    optimizer = build_optimizer(args, model)

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

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_epoch(train_loader,epoch,optimizer,model,criterion,lr_scheduler,args)
        validate(val_loader,criterion,model,width,depth)


def train_epoch(train_loader,epoch,optimizer,model,criterion,lr_scheduler,args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    logging.info('Training lr {}'.format(optimizer.param_groups[0]['lr']))

    model.train()
    for batch_idx, (images, target) in enumerate(train_loader):
        images = images.to(local_rank, non_blocking=True)
        target = target.to(local_rank, non_blocking=True)

        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

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

        # Print information about the training phase
        if batch_idx % (args.n_iters_per_epoch // 4) == 0:
            progress.display(batch_idx, logging)


def validate(val_loader,criterion,model,width,depth):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
                    len(val_loader),
                    [losses, top1, top5],
                    prefix='Test: ')

    with torch.no_grad():
        resolution = [224,224]
        model.module.set_active_subnet(resolution, width, depth)
        subnet = model.module.get_active_subnet()
        subnet.to(local_rank)
        subnet.eval()

        for batch_idx, (images, target) in enumerate(val_loader):
            images = images.to(local_rank, non_blocking=True)
            target = target.to(local_rank, non_blocking=True)

            output = subnet(images)
            loss = criterion(output, target).item()
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.size(0)

            corr1, corr5, loss = acc1 * batch_size, acc5 * batch_size, loss * batch_size
            stats = torch.tensor([corr1, corr5, loss, batch_size], device=local_rank)
            dist.barrier()  # synchronizes all processes
            dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            corr1, corr5, loss, batch_size = stats.tolist()
            acc1, acc5, loss = corr1 / batch_size, corr5 / batch_size, loss/batch_size

            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)
            losses.update(loss, batch_size)

            if batch_idx % 4 == 0:
                progress.display(batch_idx, logging)

        logging.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}, Top1: {top1.sum}/{top1.count}'
                .format(top1=top1, top5=top5))


if __name__ == "__main__":
    main()
