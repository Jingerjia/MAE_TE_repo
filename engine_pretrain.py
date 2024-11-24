# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
from transformer_engine import pytorch as te_pt


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    torch.cuda.nvtx.range_push('model_train')
    model.train(True)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push('MetricLogger')
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    torch.cuda.nvtx.range_pop()
    # set model epoch
    model.epoch = epoch
    torch.cuda.nvtx.range_push('metric_logger.log_every_before_for_loop')
    for data_iter_step, (samples, _labels, _vids) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step == 0:
            torch.cuda.nvtx.range_pop()
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        #print(samples.shape)# 64x3x224x224 for img, 64x1x512x128 for audio
        torch.cuda.nvtx.range_push('sample_to_device')
        samples = samples.to(device, non_blocking=True)
        torch.cuda.nvtx.range_pop()
        
        # comment out when not debugging
        # from fvcore.nn import FlopCountAnalysis, parameter_count_table
        # if data_iter_step == 1:
        #     flops = FlopCountAnalysis(model, samples)
        #     print(flops.total())
        #     print(parameter_count_table(model))

        torch.cuda.nvtx.range_push('loss_calculating')######
        
        use_fp8 = False
        torch.cuda.nvtx.range_push('forward')#
        with te_pt.fp8_autocast(enabled=use_fp8):
            loss_a, _, _, _, _, _ = model(samples, mask_ratio=args.mask_ratio)
        torch.cuda.nvtx.range_pop()#
        
        loss_value = loss_a.item()
        loss_total = loss_a

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        torch.cuda.nvtx.range_push('loss_other')####
        
        torch.cuda.nvtx.range_push('loss_total_accum')#
        #loss /= accum_iter
        loss_total = loss_total / accum_iter
        torch.cuda.nvtx.range_pop()#

        torch.cuda.nvtx.range_push('loss_scaler_te')#        
        if (data_iter_step + 1) % accum_iter == 0:
            torch.cuda.nvtx.range_push('optimizer_backward')#
            loss_total.backward()
            torch.cuda.nvtx.range_pop()#

            torch.cuda.nvtx.range_push('optimizer_clip_grad')#
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            torch.cuda.nvtx.range_pop()#

            torch.cuda.nvtx.range_push('optimizer_step')#
            optimizer.step()
            torch.cuda.nvtx.range_pop()#
            

            torch.cuda.nvtx.range_push('optimizer_zero_grad')#
            optimizer.zero_grad()
            torch.cuda.nvtx.range_pop()#
        else:
            norm = None

        torch.cuda.synchronize()

        torch.cuda.nvtx.range_pop()####
    
        torch.cuda.nvtx.range_pop()######
        

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        torch.cuda.nvtx.range_pop()#

        torch.cuda.nvtx.range_push('loss_value_reduce') #
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        torch.cuda.nvtx.range_pop()#

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            torch.cuda.nvtx.range_push('log_writer') #
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            torch.cuda.nvtx.range_pop()#


    torch.cuda.nvtx.range_push('synchronize_between_processes') #
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    torch.cuda.nvtx.range_pop()#
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


