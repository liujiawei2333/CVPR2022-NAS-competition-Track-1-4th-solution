import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import utils.comm as comm
from utils.utils import code2list
from .imagenet_eval import validate_one_subnet

def validate(epoch, subnets_to_be_evaluated, data_for_bn_calibration, val_loader, model, criterion, 
            writer, args, common_args, local_rank, logger, bn_calibration=True):
    supernet = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    results = []
    top1_list = []
    top5_list = []
    pre_kt_list = []

    with torch.no_grad():
        for net_id in subnets_to_be_evaluated:
            if net_id == 'nas_min_net':
                supernet.sample_min_subnet()
            elif net_id == 'nas_max_net':
                supernet.sample_max_subnet()
            else:
                stem_base_channel = 64
                block_base_channels = [64, 128, 256, 512]
                expand_ratio_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
                width, depth = code2list(net_id, stem_base_channel, block_base_channels, expand_ratio_list)

                resolution = [224,224]
                supernet.set_active_subnet(resolution, width, depth)

            subnet = supernet.get_active_subnet()
            subnet.to(local_rank)

            # Recalculate the parameters of BN layer
            if bn_calibration:
                subnet.eval()
                subnet.reset_running_stats_for_calibration()

                # Estimate running mean and running statistics
                if dist.get_rank()  == 0 and (net_id == 'nas_min_net' or net_id == 'nas_max_net'):
                    logger.info('Calirating bn running statistics')

                for _, images in enumerate(data_for_bn_calibration):
                    images = images.to(local_rank,non_blocking=True)
                    subnet(images)  #forward only

            acc1, acc5, loss, flops, params = validate_one_subnet(epoch, val_loader, subnet, 
                        criterion, net_id, writer, args, common_args, local_rank, logger)

            if not net_id == 'nas_min_net' and not net_id == 'nas_max_net':
                pre_kt_list.append(acc1)

            top1_list.append(acc1)
            top5_list.append(acc5)

            if net_id == 'nas_min_net' or net_id == 'nas_max_net':
                summary = str({
                            'net_id': net_id,
                            'mode': 'evaluate',
                            'epoch': epoch,
                            'flops': flops,
                            'params': params,
                            'acc1': acc1,
                            'acc5': acc5,
                            'loss': loss
                })

            group = comm.reduce_eval_results(summary, local_rank)
            if dist.get_rank()  == 0 and (net_id == 'nas_min_net' or net_id == 'nas_max_net'):
                results += group
                for rec in group:
                    logger.info(rec)

    return pre_kt_list, top1_list, top5_list