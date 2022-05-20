import torch
import torch.nn as nn


def build_optimizer(args, model):
    """
        Build an optimizer from config.
    """
    no_wd_params, wd_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if ".bn" in name or ".bias" in name:
                no_wd_params.append(param)
            else:
                wd_params.append(param)
    no_wd_params = nn.ParameterList(no_wd_params)
    wd_params = nn.ParameterList(wd_params)

    weight_decay_weight = args.weight_decay_weight
    weight_decay_bn_bias = args.weight_decay_bn_bias
    base_lr = args.lr_scheduler.base_lr

    params_group = [
        {"params": wd_params, "weight_decay": float(weight_decay_weight), 'group_name':'weight'},
        {"params": no_wd_params, "weight_decay": float(weight_decay_bn_bias), 'group_name':'bn_bias'},
    ]

    if args.optimizer.method == 'sgd':
        momentum = args.optimizer.momentum
        nesterov = args.optimizer.nesterov
        optimizer = torch.optim.SGD(
            params_group,
            lr = base_lr,
            momentum = momentum,
            nesterov = nesterov,
        )
    else:
        raise ValueError(f'no optimizer {args.optimizer.method}')

    return optimizer


def build_optimizer_difflr(args, group_nums, model):

    weight_decay_weight = args.weight_decay_weight
    base_lr = args.lr_scheduler.base_lr

    if group_nums == 2:
        x2_params_id = []
        x2_params = []
        for idx in range(4):#stage0
            x2_params_id += list(map(id,model.module.blocks[0][-1][idx].parameters()))
            x2_params += model.module.blocks[0][-1][idx].parameters()
        x5_params_id = []
        x5_params = []
        for idx in range(4):#stage1
            x5_params_id += list(map(id,model.module.blocks[1][-1][idx].parameters()))
            x5_params += model.module.blocks[1][-1][idx].parameters()
        x6_params_id = []
        x6_params = []
        for idx in range(8):#stage2
            x6_params_id += list(map(id,model.module.blocks[2][-1][idx].parameters()))
            x6_params += model.module.blocks[2][-1][idx].parameters()
        for idx in range(4):#stage3
            x6_params_id += list(map(id,model.module.blocks[3][-1][idx].parameters()))
            x6_params += model.module.blocks[3][-1][idx].parameters()
        normal_params = filter(lambda p:id(p) not in x2_params_id+x5_params_id+x6_params_id,model.module.parameters())

        params_group = [
            {"params": x2_params, "weight_decay": float(weight_decay_weight),"lr":base_lr*2},
            {"params": x5_params, "weight_decay": float(weight_decay_weight),"lr":base_lr*5},
            {"params": x6_params, "weight_decay": float(weight_decay_weight),"lr":base_lr*6},
            {"params": normal_params, "weight_decay": float(weight_decay_weight)}
        ]

    elif group_nums == 3:
        div4_params_id = []
        div4_params = []
        for idx in range(4):#stage1
            div4_params_id += list(map(id,model.module.blocks[1][0][idx].parameters()))
            div4_params += model.module.blocks[1][0][idx].parameters()
        div5_params_id = []
        div5_params = []
        for idx in range(8):#stage2
            div5_params_id += list(map(id,model.module.blocks[2][0][idx].parameters()))
            div5_params += model.module.blocks[2][0][idx].parameters()
        for idx in range(4):#stage3
            div5_params_id += list(map(id,model.module.blocks[3][0][idx].parameters()))
            div5_params += model.module.blocks[3][0][idx].parameters()
        normal_params = filter(lambda p:id(p) not in div4_params_id+div5_params_id,model.module.parameters())

        params_group = [
            {"params": div4_params, "weight_decay": float(weight_decay_weight),"lr":base_lr/4},
            {"params": div5_params, "weight_decay": float(weight_decay_weight),"lr":base_lr/5},
            {"params": normal_params, "weight_decay": float(weight_decay_weight)}
        ]
    
    elif group_nums == 4:
        div3_params_id = []
        div3_params = []
        for idx in range(4):#stage1
            div3_params_id += list(map(id,model.module.blocks[1][0][idx].parameters()))
            div3_params += model.module.blocks[1][0][idx].parameters()
        div4_params_id = []
        div4_params = []
        for idx in range(8):#stage2
            div4_params_id += list(map(id,model.module.blocks[2][0][idx].parameters()))
            div4_params += model.module.blocks[2][0][idx].parameters()
        for idx in range(4):#stage3
            div4_params_id += list(map(id,model.module.blocks[3][0][idx].parameters()))
            div4_params += model.module.blocks[3][0][idx].parameters()
        normal_params = filter(lambda p:id(p) not in div3_params_id+div4_params_id,model.module.parameters())

        params_group = [
            {"params": div3_params, "weight_decay": float(weight_decay_weight),"lr":base_lr/3},
            {"params": div4_params, "weight_decay": float(weight_decay_weight),"lr":base_lr/4},
            {"params": normal_params, "weight_decay": float(weight_decay_weight)}
        ]

    elif group_nums == 5:
        div2_params_id = []
        div2_params = []
        for idx in range(4):#stage1
            div2_params_id += list(map(id,model.module.blocks[1][0][idx].parameters()))
            div2_params += model.module.blocks[1][0][idx].parameters()
        div3_params_id = []
        div3_params = []
        for idx in range(8):#stage2
            div3_params_id += list(map(id,model.module.blocks[2][0][idx].parameters()))
            div3_params += model.module.blocks[2][0][idx].parameters()
        for idx in range(4):#stage3
            div3_params_id += list(map(id,model.module.blocks[3][0][idx].parameters()))
            div3_params += model.module.blocks[3][0][idx].parameters()
        normal_params = filter(lambda p:id(p) not in div2_params_id+div3_params_id,model.module.parameters())

        params_group = [
            {"params": div2_params, "weight_decay": float(weight_decay_weight),"lr":base_lr/2},
            {"params": div3_params, "weight_decay": float(weight_decay_weight),"lr":base_lr/3},
            {"params": normal_params, "weight_decay": float(weight_decay_weight)}
        ]

    elif group_nums == 6:
        div2_params_id = []
        div2_params = []
        for idx in range(8):#stage2
            div2_params_id += list(map(id,model.module.blocks[2][0][idx].parameters()))
            div2_params += model.module.blocks[2][0][idx].parameters()
        for idx in range(4):#stage3
            div2_params_id += list(map(id,model.module.blocks[3][0][idx].parameters()))
            div2_params += model.module.blocks[3][0][idx].parameters()
        normal_params = filter(lambda p:id(p) not in div2_params_id,model.module.parameters())

        params_group = [
            {"params": div2_params, "weight_decay": float(weight_decay_weight),"lr":base_lr/2},
            {"params": normal_params, "weight_decay": float(weight_decay_weight)}
        ]

    if args.optimizer.method == 'sgd':
        momentum = args.optimizer.momentum
        nesterov = args.optimizer.nesterov
        optimizer = torch.optim.SGD(
            params_group,
            lr = base_lr,
            momentum = momentum,
            nesterov = nesterov
        )
    else:
        raise ValueError(f'no optimizer {args.optimizer.method}')

    return optimizer
