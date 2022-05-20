from torch.nn.parallel import DistributedDataParallel as DDP
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

from data.DatasetLoader_imagenet import get_val_dataset_loader_imagenet
from modules.dynamic_model_few_shot import SuperNetDependTotal
from utils.utils import code2list, load_model, build_args_and_env

parser = argparse.ArgumentParser(description='subnet test')
parser.add_argument('--train-config-file', default='./configs/train_supernet_models.yml', type=str,
                    help='training configuration')
parser.add_argument('--common-config-file', default='./configs/common.yml', type=str,
                    help='common configuration')
parser.add_argument('--json_part', default=1, type=int)
parser.add_argument('--model_name', default='', type=str)
parser.add_argument('--epoch', default=1, type=int)
parser.add_argument('--save', default='', type=str)
parser.add_argument('--test_set',default=1,type=int)
parser.add_argument('--split_group', default=2, type=int)
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

    save = run_args.save
    epoch = run_args.epoch
    model_name = run_args.model_name
    json_part = run_args.json_part
    test_set = run_args.test_set
    split_group = run_args.split_group

    if split_group == 2:
        try:
            os.mkdir("./results/%s/%s/json_results/" % (save,split_group))
        except OSError:
            pass
    elif split_group > 2:
        try:
            os.mkdir("./results/%s/%s_from_%s/json_results/" % (save,split_group,split_group-1))
        except OSError:
            pass

    ## Json file saving path
    # Partial test set
    if test_set == 0:
        if split_group == 2:
            try:
                os.mkdir("./results/%s/%s/json_results/epoch%d/" % (save,split_group,epoch))
            except OSError:
                pass
        elif split_group > 2:
            try:
                os.mkdir("./results/%s/%s_from_%s/json_results/epoch%d/" % (save,split_group,split_group-1,epoch))
            except OSError:
                pass
    # Full test set
    elif test_set == 1:
        if split_group == 2:
            try:
                os.mkdir("./results/%s/%s/json_results/epoch%d_all_test/" % (save,split_group,epoch))
            except OSError:
                pass
        elif split_group > 2:
            try:
                os.mkdir("./results/%s/%s_from_%s/json_results/epoch%d_all_test/" % (save,split_group,split_group-1,epoch))
            except OSError:
                pass

    # Supernet to load
    if split_group == 2:

        common_args.resume = './results/' + save + '/' + str(split_group) + '/' + model_name + '.pth'
    elif split_group > 2:
        common_args.resume = './results/' + save + '/' + str(split_group) + '_from_' + str(split_group-1) + '/' + model_name + '.pth'
    print(common_args.resume)

    # Initial json file
    # Partial test set
    if test_set == 0:
        common_args.json_file = './json_file/' + 'part' + str(json_part) + '.json'
    # Full test set
    elif test_set == 1:
        common_args.json_file = './json_file/json8/' + 'part' + str(json_part) + '.json'
    print(common_args.json_file)

    # Save the json file
    # Partial test set
    if test_set == 0:
        if split_group == 2:
            common_args.save_file_top1 = './results/' + save + '/' + str(split_group) + '/json_results/epoch' + str(epoch) + '/top1_json'+str(json_part) + '.json'
            common_args.save_file_top5 = './results/' + save + '/' + str(split_group) + '/json_results/epoch' + str(epoch) + '/top5_json'+str(json_part) + '.json'
        elif split_group > 2:
            common_args.save_file_top1 = './results/' + save + '/' + str(split_group) + '_from_' + str(split_group-1) + '/json_results/epoch' + str(epoch) + '/top1_json'+str(json_part) + '.json'
            common_args.save_file_top5 = './results/' + save + '/' + str(split_group) + '_from_' + str(split_group-1) + '/json_results/epoch' + str(epoch) + '/top5_json'+str(json_part) + '.json'
    # Full test set
    elif test_set == 1:
        if split_group == 2:
            common_args.save_file_top1 = './results/' + save + '/' + str(split_group) + '/json_results/epoch' + str(epoch) + '_all_test' + '/top1_json'+str(json_part) + '.json'
            common_args.save_file_top5 = './results/' + save + '/' + str(split_group) + '/json_results/epoch' + str(epoch) + '_all_test' + '/top5_json'+str(json_part) + '.json'
        elif split_group > 2:
            common_args.save_file_top1 = './results/' + save + '/' + str(split_group) + '_from_' + str(split_group-1) + '/json_results/epoch' + str(epoch) + '_all_test' + '/top1_json'+str(json_part) + '.json'
            common_args.save_file_top5 = './results/' + save + '/' + str(split_group) + '_from_' + str(split_group-1) + '/json_results/epoch' + str(epoch) + '_all_test' + '/top5_json'+str(json_part) + '.json'

    stem_base_channel = 64
    block_base_channels = [64, 128, 256, 512]
    expand_ratio_list = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

    # Build model
    model = SuperNetDependTotal(common_args.n_classes,"train_supernet", dynamic_resolution=False, split_group=split_group)
    model = load_model(model, common_args.resume)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)#跨卡同步BN
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

    # Build dataloader
    args.batch_size = 256
    val_loader = get_val_dataset_loader_imagenet(test_set, args, common_args)
    args.batch_size = 128
    val_loader_for_bn = get_val_dataset_loader_imagenet(test_set, args, common_args)

    # Save a small part of the data to prepare for the recalculation of BN layer parameters in the validate stage
    data_for_bn_calibration = []

    for batch_idx, (images, _) in enumerate(val_loader_for_bn):
        # Full test set
        if test_set == 1:
            if batch_idx <= (common_args.post_bn_calibration_batch_num):
                data_for_bn_calibration.append(images)
            else:
                break
        # Partial test set
        elif test_set == 0:
            if batch_idx <= (common_args.post_bn_calibration_batch_num // 10):
                data_for_bn_calibration.append(images)
            else:
                break

    with open(common_args.json_file) as json_file:
        json_file = json.load(json_file)

    json_out_dict_top1 = dict()
    json_out_dict_top5 = dict()
    for key_index, json_key in enumerate(json_file.keys()):

        json_out_dict_top1[json_key] = dict()
        json_out_dict_top5[json_key] = dict()
        arch_acc = json_file[json_key]

        arch_code_str = arch_acc["arch"]
        json_out_dict_top1[json_key]["arch"] = arch_code_str
        json_out_dict_top5[json_key]["arch"] = arch_code_str

        width, depth = code2list(arch_code_str, stem_base_channel, block_base_channels, expand_ratio_list)
        st = time.time()

        resolution = (224,224)
        model.module.set_active_subnet(resolution, width, depth)
        subnet = model.module.get_active_subnet()
        subnet.to(local_rank)
        subnet.eval()

        # Recalculate the parameters of BN layer
        subnet.reset_running_stats_for_calibration()
        with torch.no_grad():
            for _, images in enumerate(data_for_bn_calibration):
                images = images.to(local_rank, non_blocking=True)
                subnet(images)  #forward only

        subnet.eval()
        with torch.no_grad():

            from evaluate.imagenet_eval import validate_one_subnet_during_test
            top1, top5 = validate_one_subnet_during_test(val_loader, subnet, common_args, local_rank)

        json_out_dict_top1[json_key]["acc"] = top1
        json_out_dict_top5[json_key]["acc"] = top5
        if dist.get_rank()  == 0:
            logging.info(f'{key_index + 1}/{len(json_file.keys())},top1:{top1}, top5:{top5}, time cost:{time.time() - st}s'
                .format(top1=top1))

        with open(common_args.save_file_top1, "w") as f:
            json.dump((json_out_dict_top1), f)
        with open(common_args.save_file_top5, "w") as f:
            json.dump((json_out_dict_top5), f)


if __name__ == "__main__":
    main()
