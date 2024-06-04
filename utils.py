import torch
import torch.nn as nn
import numpy as np

def define_scheduler_lambda(warmup, constant, cooldown, min_lambda = 1e-2, steps_per = 1, start_lambda = None):
    assert (warmup % steps_per == 0) and (constant % steps_per == 0) and (cooldown % steps_per == 0), "steps_per must divide warmup, constant, and cooldown"
    if start_lambda is None:
        start_lambda = min_lambda

    def scheduler_lambda(epoch):
        if epoch < warmup:
            factor = np.log(1/start_lambda) / (warmup/steps_per)
            return start_lambda * np.exp(factor * (epoch // steps_per))
        elif epoch < warmup + constant:
            return 1.0
        elif epoch < warmup + constant + cooldown:
            factor = np.log(min_lambda) / (cooldown/steps_per)
            return 1 * np.exp(factor * (epoch - warmup - constant)//steps_per)
        else:
            return min_lambda
        
    return scheduler_lambda