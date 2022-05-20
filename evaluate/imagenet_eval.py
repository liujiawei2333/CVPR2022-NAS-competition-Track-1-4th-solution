import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import time

from utils.progress import AverageMeter, ProgressMeter, accuracy
from utils.flops_counter import count_net_flops_and_params

def log_helper(summary, logger=None):
    if logger:
        logger.info(summary)
    else:
        print(summary)

def validate_one_subnet(epoch, val_loader, subnet, criterion, net_id, writer, 
            args, common_args, local_rank, logger=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    progress = ProgressMeter(
                len(val_loader),
                [batch_time, losses, top1, top5],
                prefix='Test: ')

    if dist.get_rank()  == 0 and (net_id == 'nas_min_net' or net_id == 'nas_max_net'):
        log_helper('evaluating...', logger)
    #evaluation
    end = time.time()

    subnet.to(local_rank)
    subnet.eval() # freeze again all running stats

    for batch_idx, (images, target) in enumerate(val_loader):
        images = images.to(local_rank, non_blocking=True)
        target = target.to(local_rank, non_blocking=True)

        # compute output
        output = subnet(images)
        loss = criterion(output, target).item()
        # measure accuracy
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

        if net_id == 'nas_min_net' and writer is not None:
            writer.add_scalar('val/min_subnet/'+'loss', losses.get_avg(), epoch)
            writer.add_scalar('val/min_subnet/'+'top1', top1.get_avg(), epoch)
            writer.add_scalar('val/min_subnet/'+'top5', top5.get_avg(), epoch)
        elif net_id == 'nas_max_net' and writer is not None:
            writer.add_scalar('val/max_subnet/'+'loss', losses.get_avg(), epoch)
            writer.add_scalar('val/max_subnet/'+'top1', top1.get_avg(), epoch)
            writer.add_scalar('val/max_subnet/'+'top5', top5.get_avg(), epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)

        if dist.get_rank()  == 0 and (net_id == "nas_max_net" or net_id == "nas_max_net"):
            if batch_idx % (args.n_iters_per_epoch // 4) == 0:
                progress.display(batch_idx, logger)

    if dist.get_rank()  == 0 and (net_id == "nas_max_net" or net_id == "nas_max_net"):
        log_helper(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}, Top1: {top1.sum}/{top1.count}'
            .format(top1=top1, top5=top5), logger)

    data_shape = (1, 3, 224, 224)
    flops, params = count_net_flops_and_params(subnet, data_shape)

    return float(top1.avg), float(top5.avg), float(losses.avg), flops, params

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    #rt = tensor
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def validate_one_subnet_during_test(val_loader, subnet, common_args, local_rank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    subnet.to(local_rank)
    subnet.eval() # freeze again all running stats

    for _, (images, target) in enumerate(val_loader):
        images = images.to(local_rank, non_blocking=True)
        target = target.to(local_rank, non_blocking=True)

        # compute output
        output = subnet(images)
        # measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.size(0)

        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)

    del acc1, acc5, output, subnet, batch_size

    return float(top1.avg), float(top5.avg)