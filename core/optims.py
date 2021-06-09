# --------------------------------------------------------
# RG-UNIT
# Written by Yahui Liu (yahui.liu@unitn.it)
# --------------------------------------------------------

import torch
from torch import optim

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler

import copy
from munch import Munch

def build_scheduler(
    optimizer,
    total_epochs, 
    n_iter_per_epoch,
    scheduler_type='cosine',
    warmup_epochs=10,
    decay_epochs=20,
    min_lr=1e-6,
    warmup_lr=1e-6,
    decay_rate=0.1
):
    num_steps    = int(total_epochs * n_iter_per_epoch)
    warmup_steps = int(warmup_epochs * n_iter_per_epoch)
    decay_steps  = int(decay_epochs * n_iter_per_epoch)

    lr_scheduler = None
    if scheduler_type == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif scheduler_type == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=decay_rate,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


def build_optimizer(
    model,
    opt_type='adamw', # adamw, sgd
    base_lr=1e-4
):
    opt_lower = opt_type.lower()

    optimizer = None
    params = model.parameters()
    if opt_lower == "sgd":
        optimizer = optim.SGD(
            [p for p in params if p.requires_grad], 
            momentum=0.9, 
            nesterov=True,
            lr=base_lr, 
            weight_decay=1e-4
        )
    else:
        optimizer = optim.AdamW(
            [p for p in params if p.requires_grad],
            eps=1e-8, 
            betas=(0.9, 0.999),
            lr=base_lr, 
            weight_decay=1e-4
        )
    return optimizer

