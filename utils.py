import torch
import torch.nn as nn
import numpy as np

def define_scheduler_lambda(warmup, constant, cooldown, min_lambda = 1e-2, start_lambda = None):
    if start_lambda is None:
        start_lambda = min_lambda

    def scheduler_lambda(epoch):
        if epoch < warmup:
            factor = np.log(1/start_lambda) / warmup
            return start_lambda * np.exp(factor * epoch)
        elif epoch < warmup + constant:
            return 1.0
        elif epoch < warmup + constant + cooldown:
            factor = np.log(min_lambda) / cooldown
            return 1 * np.exp(factor * (epoch - warmup - constant))
        else:
            return min_lambda
        
    return scheduler_lambda