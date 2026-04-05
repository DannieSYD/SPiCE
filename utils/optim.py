import math
import torch.optim as optim
from dataclasses import asdict


def get_optimizer(model_parameters, config):
    optimizers = {
        'Adam': lambda: optim.Adam(
            model_parameters, lr=config.learning_rate, weight_decay=config.weight_decay),
        'AdamW': lambda: optim.AdamW(
            model_parameters, lr=config.learning_rate, weight_decay=config.weight_decay, **asdict(config.adamw)),
    }
    return optimizers[config.optimizer]()


def get_scheduler(optimizer, config, train_dataset=None, world_size=None):
    schedulers = {
        'ReduceLROnPlateau': lambda: optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **asdict(config.reduce_lr_on_plateau)),
        'CosineAnnealingLR': lambda: optim.lr_scheduler.CosineAnnealingLR(
            optimizer, **asdict(config.cosine_annealing_lr)),
        'LinearWarmupCosineAnnealingLR': lambda: optim.lr_scheduler.LinearLR(
            optimizer, **asdict(config.linear_warmup_cosine_annealing_lr)),
        'OneCycleLR': lambda: optim.lr_scheduler.OneCycleLR(
            optimizer, **asdict(config.one_cycle_lr)),
        'LambdaLR': lambda: get_lr_lambda_scheduler(optimizer, config, train_dataset, world_size),
    }
    return schedulers[config.scheduler]() if config.scheduler else None


class CosineLRLambda:
    def __init__(self, scheduler_params):
        self.warmup_epochs = scheduler_params['warmup_epochs']
        self.lr_warmup_factor = scheduler_params['warmup_factor']
        self.max_epochs = scheduler_params['epochs']
        self.lr_min_factor = scheduler_params['lr_min_factor']

    def __call__(self, current_step):
        if current_step <= self.warmup_epochs:
            alpha = current_step / float(self.warmup_epochs)
            return self.lr_warmup_factor * (1.0 - alpha) + alpha
        else:
            if current_step >= self.max_epochs:
                return self.lr_min_factor
            lr_scale = self.lr_min_factor + 0.5 * (1 - self.lr_min_factor) * (
                1 + math.cos(math.pi * (current_step / self.max_epochs)))
            return lr_scale


def get_lr_lambda_scheduler(optimizer, config, train_dataset, world_size):
    assert config.scheduler == 'LambdaLR'
    assert config.lambda_lr.lambda_type == 'cosine'

    num_samples = len(train_dataset)
    batch_size = config.batch_size * 20 // world_size
    num_steps_per_epoch = num_samples // batch_size
    if num_steps_per_epoch == 0:
        num_steps_per_epoch = 1
    num_steps = num_steps_per_epoch * config.num_epochs

    scheduler_params = {
        'warmup_factor': config.lambda_lr.warmup_factor,
        'warmup_epochs': config.lambda_lr.warmup_epochs * num_steps_per_epoch,
        'epochs': num_steps,
        'lr_min_factor': config.lambda_lr.lr_min_factor
    }

    lr_lambda = CosineLRLambda(scheduler_params)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
