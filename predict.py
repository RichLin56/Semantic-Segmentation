#!/usr/bin/env python

"""predict.py: Pipeline for predicting images on trained models."""

__author__ = "Richard Lindenpuetz"
__email__ = "richard.lindenpuetz@rwth-aachen.de"
__license__ = "MIT"
__version__ = "1.0.0"

import argparse
import numpy as np
import os

import torch

import log.logging_config
import models
import utils.builder
import utils.configuration
import utils.loss_functions
import utils.predicter
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

__LOGGER.info('***Starting "Semantic-Segmentation" version %s***' %
              __version__)
cfg = utils.configuration.load_config_json(
    args.config_path, config_group='prediction')
output_dir = os.path.join(cfg['output_dir'], utils.misc.time_now())
os.makedirs(output_dir, exist_ok=True)
assert os.path.isdir(
    cfg['input_dir']), '{} does not exist'.format(cfg['input_dir'])
input_dir = cfg['input_dir']


# Define device
device = torch.device(cfg['gpu_id'] if (torch.cuda.is_available() and cfg['gpu_id'] in range(torch.cuda.device_count()))  else "cpu")
__LOGGER.info('Setting up device, using {}...'.format(device))
# Setup augmentation & data processing
__LOGGER.info('Setting up data processing...')
data_processing = utils.configuration.load_funcs_by_name(
    module=utils.augmentation, config_dict=cfg['data_processing'])

# Setup model
__LOGGER.info('Setting up model...')
model = utils.builder.ModelBuilder(
    group_cfg=cfg['network'], package=models, device=device)

# Prediction
__LOGGER.info('Starting prediction...')
predicter = utils.predicter.Predicter(model=model, 
                                      output_dir=output_dir, 
                                      data_processing=data_processing, 
                                      device=device)
for file in os.listdir(input_dir):
    file = os.path.join(input_dir, file)
    predicter.predict(file)

# Close and move logger to output_dir
log.logging_config.move_log_file(
    __LOGGER.handlers[0].baseFilename, os.path.join(output_dir, 'info.log'))




"""
@history
__version__ = "1.0.0" -> basic functionality
"""
