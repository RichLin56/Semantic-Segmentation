#!/usr/bin/env python

"""train.py: Pipeline for training semantic segmentation models implemented in pytorch."""

__author__ = "Richard Lindenpuetz"
__email__ = "richard.lindenpuetz@rwth-aachen.de"
__license__ = "MIT"
__version__ = "1.0.0"

import os
import argparse

import torch
import log
import models
import utils.builder
import utils.configuration
import utils.loss_functions
import utils.trainer
from utils.dataset import SemSegDataLoader


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
log.logging_config.setup_logging(path_to_config=os.path.join(FILE_DIR,
                                                             'log',
                                                             'logging_config.json'))
__LOGGER = log.logging_config.get_logger(__name__)

__LOGGER.info('Parsing command line...')
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--configPath', required=True,
                    dest='config_path', help='path to the .json config file')
args = parser.parse_args()
__LOGGER.info('Checking arguments...')
assert os.path.isfile(
    args.config_path), '{} does not exist'.format(args.config_path)

__LOGGER.info('***Starting "Semantic-Segmentation Training" version %s***' %
              __version__)
cfg = utils.configuration.load_config_json(
    args.config_path, config_group='training')
output_dir = os.path.join(cfg['output_dir'], utils.misc.time_now())
os.makedirs(output_dir, exist_ok=True)

# Define device
device = torch.device(cfg['gpu_id'] if torch.cuda.is_available() else "cpu")
__LOGGER.info('Setting up device, using {}...'.format(device))
# Setup augmentation & data processing
__LOGGER.info('Setting up augmentations & data processing...')
augmentations = utils.configuration.load_funcs_by_name(
    module=utils.augmentation, config_dict=cfg['augmentation'])
data_processing = utils.configuration.load_funcs_by_name(
    module=utils.augmentation, config_dict=cfg['data_processing'])
# Setup dataset
__LOGGER.info('Setting up dataset...')
train_loader = SemSegDataLoader(
    subset='train', augmentations=augmentations, data_processing=data_processing, **cfg['data'])
val_loader = SemSegDataLoader(
    subset='val', data_processing=data_processing, **cfg['data'])
data_train = {'train': train_loader, 'val': val_loader}

if cfg['test_afterwards']:
    test_loader = SemSegDataLoader(
        subset='test', data_processing=data_processing, **cfg['data'])
    data_test = {'test': test_loader}

# Setup metric
__LOGGER.info('Setting up metric...')
val_meter = utils.misc.MetricMeter(
    metric=cfg['val_metric'], best_avg=0, comp=max)
# Setup model
__LOGGER.info('Setting up model...')
model = utils.builder.ModelBuilder(
    group_cfg=cfg['network'], package=models, device=device)
# Setup optimizer
__LOGGER.info('Setting up optimizer...')
optimizer = utils.builder.OptimizerBuilder(group_cfg=cfg['optimizer'], package=torch.optim,
                                           model=model.model)
# Setup scheduler
__LOGGER.info('Setting up scheduler...')
scheduler = utils.builder.SchedulerBuilder(group_cfg=cfg['scheduler'], module=torch.optim.lr_scheduler,
                                           optimizer=optimizer.optimizer, val_meter=val_meter)
# Setup criterion
__LOGGER.info('Setting up criterion...')
criterion = utils.builder.CriterionBuilder(group_cfg=cfg['loss_function'], module=utils.loss_functions,
                                           device=device)
# Training routine
__LOGGER.info('Starting training routine...')
trainer = utils.trainer.Trainer(model=model.model,
                                optimizer=optimizer.optimizer,
                                criterion=criterion.criterion,
                                scheduler=scheduler.scheduler,
                                num_epochs=cfg['num_epochs'],
                                val_meter=val_meter,
                                output_dir=output_dir,
                                device=device)
trainer.train(data_train)
# Test routine
if cfg['test_afterwards']:
    __LOGGER.info('Starting test routine...')
    trainer.test(data_test)
# Close and move logger to output_dir
log.logging_config.move_log_file(
    __LOGGER.handlers[0].baseFilename, os.path.join(output_dir, 'info.log'))


"""
@history
__version__ = "1.0.0" -> basic functionality
"""
