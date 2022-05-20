import torch
import math

class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, 
        optimizer,
        max_iters,
        warmup_factor = 0.001,
        warmup_iters = 1000,
        warmup_method = 'linear',
        last_epoch = -1,
        clamp_lr = 0.
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.clamp_lr = clamp_lr
        super().__init__(optimizer, last_epoch)


    def get_lr(self):
        warmup_factor = _get_warmup_factor_at_iter(
                self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [ max( self.clamp_lr if self.last_epoch > self.warmup_iters else 0.,
                      base_lr * warmup_factor * 0.5 * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
                    ) for base_lr in self.base_lrs ]

    def _compute_values(self):
        return self.get_lr()

def _get_warmup_factor_at_iter(method, iter, warmup_iters, warmup_factor):
    if iter >= warmup_iters:
        return 1.0
    if method == 'constant':
        return warmup_factor
    elif method == 'linear':
        alpha = float(iter) / float(warmup_iters)
        return warmup_factor * (1. - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))