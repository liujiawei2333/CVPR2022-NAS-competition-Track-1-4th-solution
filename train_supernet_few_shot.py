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
from solver.build import build_optimizer_difflr,build_optimizer
from solver.lr_scheduler import WarmupCosineLR
from modules.dynamic_model_few_shot import SuperNetDependTotal
from data.DatasetLoader_imagenet import get_dataset_loader_imagenet

from utils.utils import build_args_and_env
from utils.comm import create_exp_dir
from evaluate import nas_eval as nas_eval
import utils.saver as saver
import utils.comm as comm
import loss_ops as loss_ops


parser = argparse.ArgumentParser(description='few-shot weight sharing nas supernet training')
parser.add_argument('--train-config-file', default='./configs/train_supernet_models.yml', type=str,
                    help='training configuration')
parser.add_argument('--common-config-file', default='./configs/common.yml', type=str,
                    help='common configuration')
parser.add_argument('--group_nums', default=2, type=int)
parser.add_argument('--resumed', action="store_false", default=True)
parser.add_argument('--pretrained_path', default="", type=str)
parser.add_argument('--split_model_path', default="", type=str)
parser.add_argument('--resumed_split_groups', default=2, type=int)
run_args = parser.parse_args()

local_rank = int(os.environ.get('LOCAL_RANK', '-1'))
if local_rank != -1:
    torch.cuda.set_device(local_rank)


def main():
    args, common_args = build_args_and_env(run_args)
    main_worker(args, common_args)


def main_worker(args, common_args):

    # Load from the model trained in the previous few- Shot NAS phase
    if run_args.pretrained_path == "" and run_args.split_model_path != "":
        try:
            os.mkdir("./results/%s/%d_from_%d/" % (common_args.save,run_args.group_nums,run_args.resumed_split_groups))
        except OSError:
            pass
        save_path = './results/' + common_args.save + '/' + str(run_args.group_nums) + '_from_' + str(run_args.resumed_split_groups) + '/'

    # Load from the one-Shot NAS trained model
    elif run_args.pretrained_path != "" and run_args.split_model_path == "":
        try:
            os.mkdir("./results/%s/" % (common_args.save))
        except OSError:
            pass
        try:
            os.mkdir("./results/%s/%d/" % (common_args.save,run_args.group_nums))
        except OSError:
            pass
        save_path = './results/' + common_args.save + '/' + str(run_args.group_nums) + '/'

    # Create log save path
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
    args.checkpoint_save_path = os.path.join(save_path, 'checkpoint.pth.tar')

    # Create log
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    random.seed(common_args.seed)
    cudnn.benchmark = True
    torch.manual_seed(common_args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(common_args.seed)
    dist.init_process_group("nccl")

    writer = SummaryWriter(os.path.join(save_path, 'tensorboard_log'))

    # Build model
    comm.synchronize()
    model = SuperNetDependTotal(common_args.n_classes, "train_supernet", dynamic_resolution=True, split_group=run_args.group_nums)

    if run_args.resumed:
        assert (run_args.split_model_path != "") ^ (run_args.pretrained_path != "") == 1

        if run_args.group_nums == 3:
            args.lr_scheduler.base_lr = 0.002
        else:
            args.lr_scheduler.base_lr = 0.001

        if run_args.pretrained_path != "":
            model.init_weights_from_one(run_args.pretrained_path)
            logging.info("=> load one-shot weights from  '{}'".format(run_args.pretrained_path))

        if run_args.split_model_path != "":
            model.init_weights(run_args.split_model_path, run_args.resumed_split_groups)
            logging.info("=> load few-shot weights from  '{}'".format(run_args.split_model_path))

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Build loss
    criterion = loss_ops.CrossEntropyLossSmooth(args.label_smoothing).to(local_rank)
    soft_criterion = loss_ops.KLLossSoft().to(local_rank)

    # Build dataloader
    train_loader, val_loader, train_sampler = get_dataset_loader_imagenet(args,common_args)

    args.n_iters_per_epoch = len(train_loader)

    # Build optimizer and learning rate scheduler
    if run_args.group_nums == 7:
        optimizer = build_optimizer(args, model)
    else:
        optimizer = build_optimizer_difflr(args, run_args.group_nums, model)

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
            warmup_factor=warmup_lr,
            warmup_iters=args.warmup_iters,
            warmup_method=warmup_method,
            clamp_lr=clamp_lr,
        )

    # In case of interruption, load parameters and continue training
    if args.resume:
        saver.load_checkpoints(args, model, optimizer, lr_scheduler, logging)

    subnets_to_be_evaluated = {
        'nas_min_net': {},
        'nas_max_net': {},
    }

    json_file = './json_file/gt_arch30.json'# ACC of stand alone trained 30 subnets

    with open(json_file) as json_file:
        json_file = json.load(json_file)

    gt_kt_list = []
    # The subnets waiting to be evaluated include the max-subnets, the min-subnets, and 30 stand alone trained subnets
    for json_key in json_file.keys():
        subnets_to_be_evaluated[json_file[json_key]["arch"]] = {}
        gt_kt_list.append(json_file[json_key]["acc"])

    if dist.get_rank() == 0:
        logging.info(args)

    max_Kendallta = -999

    for epoch in range(args.start_epoch, args.epochs):

        train_sampler.set_epoch(epoch)

        acc1, acc5, data_for_bn_calibration = train_epoch(epoch, model, train_loader, optimizer, criterion,
            writer, common_args, args, soft_criterion=soft_criterion, lr_scheduler=lr_scheduler)

        pre_kt_list, top1_list, top5_list = validate(epoch, subnets_to_be_evaluated, data_for_bn_calibration,
                    val_loader, model, criterion, writer, args, common_args)

        Kendallta, _ = kendalltau(gt_kt_list, pre_kt_list)

        writer.add_scalar('val/Kendallta', Kendallta, epoch)
        writer.add_scalar('valid/top1', sum(top1_list) / len(top1_list), epoch)
        writer.add_scalar('valid/top5', sum(top5_list) / len(top5_list), epoch)

        if dist.get_rank() == 0:
            logging.info('Kendallta {}'.format(Kendallta))
            logging.info('total_net_top1 {}'.format(sum(top1_list) / len(top1_list)))
            logging.info('total_net_top5 {}'.format(sum(top5_list) / len(top5_list)))

            # In case of interruption, save parameters and continue training
            saver.save_checkpoint(
                args.checkpoint_save_path,
                model,
                optimizer,
                lr_scheduler,
                args,
                epoch,
            )

            # Save models
            torch.save(model.state_dict(), os.path.join(save_path,
                    str(Kendallta) + "_" + str(epoch) + ".pth"))

            if Kendallta > max_Kendallta or epoch == 0:
                max_Kendallta = Kendallta
                torch.save(model.state_dict(), os.path.join(save_path, "supernet_max_kt.pth"))


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

    writer.add_scalar('learning_rate/lr', optimizer.param_groups[-1]['lr'], epoch)#-1代表参数组的最后一个即normal_params
    if local_rank == 0:
        logging.info('Training lr {}'.format(optimizer.param_groups[-1]['lr']))

    model.train()
    end = time.time()

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
            else:
                model.module.sample_active_subnet()

            output = model(images)
            if soft_criterion:
                loss = soft_criterion(output, soft_logits)
            else:
                assert not args.inplace_distill
                loss = criterion(output, target)
            loss.backward()

        if getattr(args, 'grad_clip_value', None):
            torch.nn.utils.clip_grad_value_(model.parameters(), args.grad_clip_value)

        optimizer.step()

        # Accuracy measured on the local batch
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        corr1, corr5, loss = acc1 * args.batch_size, acc5 * args.batch_size, loss.item() * args.batch_size  # just in case the batch size is different on different nodes
        stats = torch.tensor([corr1, corr5, loss, args.batch_size], device=local_rank)
        dist.barrier()  # synchronizes all processes
        dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        corr1, corr5, loss, batch_size = stats.tolist()
        acc1, acc5, loss = corr1 / batch_size, corr5 / batch_size, loss / batch_size
        losses.update(float(loss), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

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

    return nas_eval.validate(epoch, subnets_to_be_evaluated, data_for_bn_calibration, val_loader, model, criterion,
                            writer, args, common_args, local_rank, logging, bn_calibration=True)


if __name__ == '__main__':
    main()
