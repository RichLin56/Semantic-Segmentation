#!/usr/bin/env python

"""test.py: Pipeline for testing semantic segmentation models implemented in pytorch."""

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

__LOGGER.info('***Starting "Semantic-Segmentation Testing" version %s***' %
              __version__)
cfg = utils.configuration.load_config_json(
    args.config_path, config_group='testing')
output_dir = os.path.join(cfg['output_dir'], utils.misc.time_now())
os.makedirs(output_dir, exist_ok=True)

# Define device
device = torch.device(cfg['gpu_id'] if torch.cuda.is_available() else "cpu")
__LOGGER.info('Setting up device, using {}...'.format(device))
# Setup augmentation & data processing
__LOGGER.info('Setting up data processing...')
data_processing = utils.configuration.load_funcs_by_name(
    module=utils.augmentation, config_dict=cfg['data_processing'])
# Setup dataset
__LOGGER.info('Setting up dataset...')
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

# Testing routine
__LOGGER.info('Starting test routine...')
trainer = utils.trainer.Trainer(model=model.model,
                                optimizer=None,
                                criterion=None,
                                scheduler=None,
                                num_epochs=None,
                                val_meter=val_meter,
                                output_dir=output_dir,
                                device=device)

trainer.test(data_test)
# Close and move logger to output_dir
log.logging_config.move_log_file(
    __LOGGER.handlers[0].baseFilename, os.path.join(output_dir, 'info.log'))


"""
@history
__version__ = "1.0.0" -> basic functionality
"""
