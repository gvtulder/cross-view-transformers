# Cross-view transformers for multi-view analysis of unregistered medical images
# Copyright (C) 2021 Gijs van Tulder / Radboud University, the Netherlands
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
import argparse
import collections
import os
import sys
import time
import numpy as np
import json
import h5py
import matplotlib.pyplot as plt
import re
import sklearn.metrics
import scipy.special
import tqdm

import torch
import torch.nn as nn
import torch.cuda.amp
import torch.utils
import torch.utils.tensorboard

import net
import ddsm_data
import chexpert_data
import util
import two_view_attention


parser = argparse.ArgumentParser()
parser.add_argument('--train-data', metavar='HDF5', nargs='+',
                    help='training samples (hdf5)')
parser.add_argument('--val-data', metavar='HDF5', nargs='+',
                    help='validation samples (hdf5)')
parser.add_argument('--test-data', metavar='HDF5', nargs='+',
                    help='test samples (hdf5)')
parser.add_argument('--epochs', metavar='N', type=int, default=100)
parser.add_argument('--lr', metavar='LR', type=float, default=0.001)
parser.add_argument('--lr-schedule', metavar='SCHEDULER',
                    choices=['StepLR', 'CosineAnnealingLR'],
                    help='use a learning rate scheduler')
parser.add_argument('--lr-schedule-step-size', metavar='STEP', type=int,
                    help='step size for the learning rate scheduler')
parser.add_argument('--lr-schedule-gamma', metavar='GAMMA', type=float,
                    help='gamma for the learning rate scheduler')
parser.add_argument('--lr-schedule-eta-min', metavar='LR', type=float,
                    help='the final learning rate for the annealing scheduler')
parser.add_argument('--lr-warmup-epochs', metavar='EPOCHS', type=int,
                    help='implement linear warmup for the first epochs')
parser.add_argument('--optimizer', metavar='OPTIM', choices=['Adam', 'SGD'], default='Adam',
                    help='set the optimizer')
parser.add_argument('--swa', action='store_true',
                    help='enable stochastic weight averaging')
parser.add_argument('--swa-start-epoch', metavar='EPOCH', type=int,
                    help='start stochastic weight averaging from this epoch')
parser.add_argument('--swa-lr', metavar='LR', type=float,
                    help='learning rate for stochastic weight averaging')
parser.add_argument('--freeze-batch-norm', action='store_true',
                    help='sets the model to eval() during training')
parser.add_argument('--mb-size', metavar='N', type=int, default=6)
parser.add_argument('--device', metavar='DEVICE', type=str, default='cpu')
parser.add_argument('--tensorboard-dir', metavar='DIR')
parser.add_argument('--checkpoints-dir', metavar='DIR')
parser.add_argument('--best-checkpoints-only', action='store_true',
                    help='only keep the best and final checkpoints')
parser.add_argument('--augment', choices=['flip', 'rot90', 'cropzoom', 'elastic',
                                          'rotate', 'crop20', 'coflip'], nargs='+')
parser.add_argument('--model', choices=net.models, required=True)
parser.add_argument('--pretrained', action='store_true', default=None,
                    help='use ImageNet-pretrained models')
parser.add_argument('--no-pretrained', action='store_false', dest='pretrained', default=None,
                    help='do not use ImageNet-pretrained models')
parser.add_argument('--pretrained-weights', metavar='WEIGHTS',
                    help='load the weights for a complete model')
parser.add_argument('--classes', required=True, type=int)
parser.add_argument('--tasks', type=int, default=1)
parser.add_argument('--weighted-loss', action='store_true',
                    help='use a weighted loss function based on class balance')
parser.add_argument('--label-smoothing', action='store_true',
                    help='set target labels to 0.1 and 0.9')
parser.add_argument('--data', choices=list(ddsm_data.datasets) + \
                                      list(chexpert_data.datasets), required=True)
parser.add_argument('--ddsm-views', default=['cc', 'mlo'], nargs='+')
parser.add_argument('--chexpert-views', default=['frontal', 'lateral'], nargs='+')
parser.add_argument('--view-dropout', action='store_true', default=None)
parser.add_argument('--dropout', metavar='P', type=float, default=None,
                    help='enable dropout on some models')
parser.add_argument('--attention-heads', type=int, default=None)
parser.add_argument('--attention-downsampling', type=int, default=None)
parser.add_argument('--attention-combine', choices=('add', 'add-linear',
                                                    'ln-add-linear-do',
                                                    'linear-linear',
                                                    'concatenate'), default=None)
parser.add_argument('--attention-bidirectional', action='store_true', default=None)
parser.add_argument('--attention-l1-loss', type=float, default=None,
                    help='weight for the L1 loss on the transformer coefficients')
parser.add_argument('--attention-tokens', type=int, default=None,
                    help='enables tokenization of B before attention')
parser.add_argument('--attention-token-layers', type=int, default=None,
                    help='enables tokenization of B before attention')
parser.add_argument('--attention-tokenize-a', action='store_const', const=True, default=None,
                    help='enables tokenization of A as well')
parser.add_argument('--attention-implementation', default='samplewise-directsum',
                    choices=('traditional', 'samplewise-directsum'),
                    help='seleect the two-view attention implementation')
parser.add_argument('--normalize', action='store_true', default=False)
parser.add_argument('--num-workers', metavar='N', type=int, default=0)
parser.add_argument('--autocast', action='store_true', default=False,
                    help='enable the torch.cuda.amp autocasting')
parser.add_argument('--git-commit', metavar='HASH',
                    help='can be used to store the current git commit hash')
args = parser.parse_args()
vargs = vars(args)
print(args)
print()


dtype = torch.float
device = torch.device(args.device)


# compute number of outputs
if args.classes > 2:
    classes = 3
    tasks = args.tasks
    outputs = classes * tasks
else:
    assert args.tasks == 1
    classes = 2
    tasks = 1
    outputs = 1

# compute number of input channels
if 'MNIST' in args.data:
    in_channels = 3
else:
    in_channels = 1

# additional model parameters?
model_args = {}
if args.attention_heads:
    model_args['heads'] = args.attention_heads
if args.attention_l1_loss is not None:
    model_args['attention_l1_loss'] = True
for argname in ('attention_downsampling', 'attention_combine',
                'attention_bidirectional', 'attention_tokens',
                'attention_token_layers', 'attention_tokenize_a',
                'dropout', 'view_dropout', 'pretrained'):
    if vargs[argname] is not None:
        model_args[argname] = vargs[argname]

# select attention implementation
if args.attention_implementation:
    two_view_attention.TwoViewAttentionModule.implementation = args.attention_implementation

# initialize classification network
if args.model in net.models:
    net_class = net.models[args.model]
else:
    raise Exception('Unknown net: %s' % args.model)
model = net_class(in_channels=in_channels, outputs=outputs, **model_args).to(device)
print(model)


# load pretrained weights
if args.pretrained_weights:
    print('Load pretrained weights from %s' % args.pretrained_weights)
    d = torch.load(args.pretrained_weights, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(d['model_state_dict'], strict=False)
    print('Missing keys:', missing_keys)
    print('Unexpected keys:', unexpected_keys)
    del d


# loss functions
def accuracy_fn(y_pred, y_true, ignore_index=None):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    if ignore_index is not None:
        # skip targets with this target label
        y_pred = y_pred[y_true != ignore_index]
        y_true = y_true[y_true != ignore_index]
    return np.mean(np.equal(y_pred, y_true))

def roc_auc_fn(y_score, y_true):
    y_score = y_score.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    return sklearn.metrics.roc_auc_score(y_true, y_score,
                                         multi_class='ovo', average='macro')

# optimizer
if args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
else:
    raise Exception('Unknown optimizer: %s' % args.optimizer)


# learning rate scheduler
if args.lr_schedule == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_schedule_step_size,
                                                gamma=args.lr_schedule_gamma)
elif args.lr_schedule == 'CosineAnnealingLR':
    eta_min = args.swa_lr if args.swa else (args.lr_schedule_eta_min or 0)
    T_max = args.epochs - (args.lr_warmup_epochs or 0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max,
                                                           eta_min=eta_min, verbose=True)
else:
    assert args.lr_schedule is None
    scheduler = None


# additional learning rate warmup
if args.lr_warmup_epochs is not None:
    # linear increase until args.lr_warmup_epochs
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, verbose=True,
            lr_lambda=lambda epoch: min(1.0, epoch / args.lr_warmup_epochs))



# enable stochastic weight averaging (if required)
if args.swa:
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=args.swa_lr)


# scaler
scaler = torch.cuda.amp.GradScaler(args.autocast)


# initialize loaders
def worker_init_fn(w):
    # workers need different random seeds to produce independent samples
    worker_seed = torch.utils.data.get_worker_info().seed % (2**32)
    np.random.seed(worker_seed)

datasets = {}
if hasattr(ddsm_data, args.data):
    # DDSM data
    dataset_class = ddsm_data.datasets[args.data]
    for key in ('train', 'val', 'test'):
        augmentation = args.augment if 'train' in key else []
        datasets[key] = [dataset_class(filename,
                                       augment=augmentation,
                                       normalize=args.normalize,
                                       dtype_x=dtype,
                                       dtype_y=dtype,
                                       views=args.ddsm_views)
                         for filename in vargs['%s_data' % key]]
    number_of_views = len(args.ddsm_views)
elif hasattr(chexpert_data, args.data):
    # CheXpert data
    dataset_class = chexpert_data.datasets[args.data]
    for key in ('train', 'val', 'test'):
        augmentation = args.augment if 'train' in key else []
        datasets[key] = [dataset_class(filename,
                                       augment=augmentation,
                                       dtype_x=dtype,
                                       dtype_y=dtype,
                                       views=args.chexpert_views)
                         for filename in vargs['%s_data' % key]]
    number_of_views = len(args.chexpert_views)
else:
    raise Exception('Unknown data: %s' % args.data)

# compute loss weights
if args.weighted_loss:
    # give a weight of 1 to the largest class, > 1 for the others
    class_weight = sum([d.class_freq() for d in datasets['train']]).to(torch.float)
    class_weight = class_weight.max() / class_weight
    print('Weighted loss:', class_weight)
else:
    class_weight = None

# construct objective function
if args.classes > 2:
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weight, ignore_index=99)
else:
    if class_weight is None:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weight[1])
loss_fn = loss_fn.to(device)

# construct loaders
loaders = {}
for key in ('train', 'val', 'test'):
    print('Loading %s data' % key)
    ds = datasets[key]
    if isinstance(ds, list):
        ds = torch.utils.data.ConcatDataset(ds)

    loaders[key] = torch.utils.data.DataLoader(ds, batch_size=args.mb_size,
                                               shuffle='train' in key,
                                               num_workers=args.num_workers,
                                               worker_init_fn=worker_init_fn,
                                               pin_memory=True)
    print('Dataset %s: %d samples' % (key, len(ds)))


# initalize directories
if args.tensorboard_dir:
    tensorboard_writer = torch.utils.tensorboard.SummaryWriter(args.tensorboard_dir)
    tensorboard_writer.add_text('args', json.dumps(vars(args)))
else:
    tensorboard_writer = None
if args.checkpoints_dir:
    os.makedirs(args.checkpoints_dir, exist_ok=True)

# keep track of the best validation accuracy and roc_auc found so far,
# so we can save the parameters of the best model
best_val_score = {'accuracy': None, 'roc_auc': None}


# training
# run one extra epoch for the final SWA model to learn the batch normalization parameters
for epoch in range(args.epochs + (1 if args.swa else 0)):
    if args.swa and epoch == args.epochs:
        print('Final epoch %d for SWA' % epoch)
        model = swa_model
        # see torch.optim.swa_util.update_bn
        # this run is to optimize the batch normalization parameters,
        # so we should reset them first
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                module.momentum = None
                module.num_batches_tracked *= 0
    else:
        print('Epoch %d' % epoch)

    time_start = time.time()

    all_losses = {}
    all_histograms = {}
    all_figures = {}
    epoch_class_balance_estimates = collections.defaultdict(float)
    epoch_class_balance_true = collections.defaultdict(float)

    # collect predictions for all samples in this epoch
    epoch_y = collections.defaultdict(list)
    epoch_predictions = collections.defaultdict(list)
    epoch_predicted_labels = collections.defaultdict(list)

    # training/validation
    for phase in ('train', 'val', 'test'):
        # reset counters
        losses = collections.defaultdict(float)
        histograms = collections.defaultdict(list)
        confmat = np.zeros((classes, classes), dtype='int')
        n = 0

        model.train(phase == 'train' and not args.freeze_batch_norm)
        torch.set_grad_enabled(phase == 'train')

        for (*x_views, y) in tqdm.tqdm(loaders[phase], leave='true', desc=('%-6s' % phase), disable=None, dynamic_ncols=True):
            prediction = {}
            class_loss = {}

            # convert type and move to gpu
            x_views = [x.to(device=device, dtype=dtype, non_blocking=True) for x in x_views]
            y = y.to(device=device, dtype=(torch.long if outputs > 1 else dtype), non_blocking=True)

            with torch.cuda.amp.autocast(args.autocast):
                # compute forward pass
                prediction = model(*x_views)
                if args.attention_l1_loss is not None:
                    prediction, *extra_outputs = prediction

                # compute loss
                if outputs == 1:
                    # binary classification
                    if args.label_smoothing:
                        # scale targets to 0.1 and 0.9
                        y_for_loss = y * 0.8 + 0.1
                    else:
                        y_for_loss = y
                    if prediction.ndim == 2:
                        # standard case: one output per sample
                        class_loss = loss_fn(prediction[:, 0], y_for_loss)
                        prediction_label = (prediction[:, 0].detach() > 0)
                        accuracy = accuracy_fn(prediction_label, y)
                        prediction_label = prediction_label.to(int)
                        loss = class_loss
                    else:
                        assert prediction.ndim == 3
                        # special case with multiple outputs
                        # each output has the same target
                        class_loss = loss_fn(prediction[:, :, 0], y_for_loss[:, None].expand(-1, prediction.shape[1]))
                        # the first output is the main output
                        prediction = prediction[:, 0, :]
                        prediction_label = (prediction[:, 0].detach() > 0)
                        accuracy = accuracy_fn(prediction_label, y)
                        prediction_label = prediction_label.to(int)
                        loss = class_loss
                elif tasks > 1:
                    # multi-task, multi-class, one-hot encoding
                    prediction = prediction.view(-1, classes, tasks)
                    class_loss = loss_fn(prediction, y)
                    prediction_label = torch.argmax(prediction, dim=1)
                    accuracy = accuracy_fn(prediction_label, y, ignore_index=99)
                    loss = class_loss
                else:
                    # multi-class, one-hot encoding
                    class_loss = loss_fn(prediction, y)
                    prediction_label = torch.argmax(prediction, dim=1)
                    accuracy = accuracy_fn(prediction_label, y)
                    loss = class_loss

                # add attention L1 losses if required
                if args.attention_l1_loss is not None:
                    # two outputs for bidirectional attention
                    assert len(extra_outputs) in (1, 2)
                    attention_l1_loss = sum(torch.mean(eo) for eo in extra_outputs)
                    loss = loss + args.attention_l1_loss * attention_l1_loss


            # add losses
            losses['loss/%s' % phase] += loss.item()
            losses['class_loss/%s' % phase] += class_loss.item()
            losses['accuracy/%s' % phase] += accuracy
            if args.attention_l1_loss is not None:
                losses['attention_l1_loss/%s' % phase] += attention_l1_loss.item()

            # update confmat
            for true_label in range(classes):
                for pred_label in range(classes):
                    confmat[true_label, pred_label] += torch.sum((y == true_label) * (prediction_label == pred_label)).item()

            # store predictions
            epoch_y[phase].append(y.detach().cpu().numpy())
            epoch_predictions[phase].append(prediction.detach().cpu().numpy())
            epoch_predicted_labels[phase].append(prediction_label.detach().cpu().numpy())

            # increment minibatch counter
            n += 1

            if phase == 'train' and not (args.swa and epoch == args.epochs):
                # next: compute and apply updates
                optimizer.zero_grad()
                if args.autocast:
                  scaler.scale(loss).backward()
                  scaler.step(optimizer)
                  scaler.update()
                else:
                  loss.backward()
                  optimizer.step()

            del x_views, y, loss, class_loss, prediction, prediction_label, accuracy

        # add to list of scores
        for k in losses.keys():
            all_losses[k] = losses[k] / n

        all_figures['confmat/%s' % phase] = util.plot_confmat(confmat)

    time_end = time.time()

    if args.lr_warmup_epochs and epoch < args.lr_warmup_epochs:
        # warmup during initial steps
        warmup_scheduler.step()
    elif args.swa and epoch == args.epochs:
        # apply stochastic weight averaging
        # final epoch only to learn batch normalization parameters
        pass
    elif args.swa and epoch > args.swa_start_epoch:
        # apply stochastic weight averaging
        # update the swa model
        swa_model.update_parameters(model)
        swa_scheduler.step()
    elif scheduler is not None:
        # update the learning rate using the normal scheduler
        scheduler.step()

    # compute roc_auc and f1 scores
    for phase in epoch_y:
        epoch_y[phase] = np.concatenate(epoch_y[phase], 0)
        epoch_predictions[phase] = np.concatenate(epoch_predictions[phase], 0)
        epoch_predicted_labels[phase] = np.concatenate(epoch_predicted_labels[phase], 0)

        if 'CheXpert' in args.data:
            # take negative/positive outputs only, compute softmax for each task,
            # then take the prediction for the positive class
            chexpert_pred_prob = scipy.special.softmax(epoch_predictions[phase][:, 1:, :], axis=1)[:, 1, :]
            # compute ROC-AUC for each task
            for task_idx, task in enumerate(chexpert_data.CheXpertDataset.TASKS):
                # find only targets with a negative (1) or positive (2) label
                sample_mask = np.logical_or(epoch_y[phase][:, task_idx] == 1, epoch_y[phase][:, task_idx] == 2)
                if len(np.unique(epoch_y[phase][sample_mask, task_idx])) > 1:
                    all_losses['roc_auc_task/%s/%s' % (phase, task)] = \
                        sklearn.metrics.roc_auc_score(epoch_y[phase][sample_mask, task_idx] > 1,
                                                      chexpert_pred_prob[sample_mask, task_idx])

        if outputs == 1:
            # this only works for binary classification problems
            all_losses['roc_auc/%s' % phase] = sklearn.metrics.roc_auc_score(epoch_y[phase] > 0.5, epoch_predictions[phase][:, 0].flatten())
            all_figures['roc_curve/%s' % phase] = util.plot_roc_curve(epoch_y[phase] > 0.5, epoch_predictions[phase][:, 0].flatten())
            all_losses['f1_score/%s' % phase] = sklearn.metrics.f1_score(epoch_y[phase] > 0.5, epoch_predicted_labels[phase])
        all_histograms['predictions/%s' % phase] = epoch_predictions[phase][:, 0].flatten()

    # is this the best model so far?
    for metric in best_val_score:
        if ('%s/val' % metric) not in all_losses:
            continue
        if best_val_score[metric] is None or all_losses['%s/val' % metric] > best_val_score[metric]:
            best_val_score[metric] = all_losses['%s/val' % metric]
            # save weights
            if args.checkpoints_dir:
                torch.save({
                    'epoch': epoch,
                    'vargs': vargs,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': all_losses,
                }, '%s/best_val_%s.tar' % (args.checkpoints_dir, metric))

    # save weights for final epoch and every fifth epoch
    if args.checkpoints_dir and ((epoch >= args.epochs - 1) or
                                 (epoch % 5 == 0 and not args.best_checkpoints_only)):
        torch.save({
            'epoch': epoch,
            'vargs': vargs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': all_losses,
          }, '%s/%03d.tar' % (args.checkpoints_dir, epoch))


    for k, v in sorted(all_losses.items()):
        print('  %s: %f' % (k, v))
    print('  time: %0.2fs' % (time_end - time_start))

    if tensorboard_writer:
        for k, v in all_losses.items():
            tensorboard_writer.add_scalar(k, v, epoch)
        for k, v in all_histograms.items():
            tensorboard_writer.add_histogram(k, v, epoch)
        for k, v in all_figures.items():
            tensorboard_writer.add_figure(k, v, epoch)
        tensorboard_writer.add_scalar('time per epoch', time_end - time_start, epoch)

    plt.close('all')

