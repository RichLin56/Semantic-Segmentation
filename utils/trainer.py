#!/usr/bin/env python

"""trainer.py: Class to manage a trainings routine for semantic segmentation."""

__author__ = "Richard Lindenpuetz"
__email__ = "richard.lindenpuetz@rwth-aachen.de"
__license__ = "MIT"
__version__ = "1.0.0"

import torch
from torch.utils.tensorboard import SummaryWriter

import utils
from log.logging_config import get_logger


class Trainer(object):
    def __init__(self, model, optimizer, criterion,
                 num_epochs, val_meter, output_dir,
                 device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.val_meter = val_meter
        self.output_dir = output_dir
        self.device = device
        self.scheduler = scheduler
        self.loss_meter = utils.misc.AverageMeter()
        self.__LOGGER = get_logger(name=__name__)
        # Setup Summary Writer
        self.__LOGGER.info('Setting up SummaryWriter (Tensorboard), \
					  file can be found here: {}'.format(self.output_dir))
        self.summarywriter = SummaryWriter(self.output_dir)

    def train(self, data_loader: dict):

        for epoch in range(self.num_epochs):
            self.__LOGGER.info('-' * 10)
            self.__LOGGER.info('Epoch {}/{}'.format(epoch+1, self.num_epochs))
            for phase in data_loader.keys():
                if phase == 'train':
                    self.model.train(True)
                else:
                    self.model.eval()
                self.loss_meter.reset()
                self.val_meter.reset()
                for i, (images, labels) in enumerate(data_loader[phase]):
                    # Sent images and labels to device
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    # Forward, track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(images)
                        # Measure loss
                        loss = self.criterion(outputs, labels)
                        self.loss_meter.update(loss.item(), images.shape[0])
                        self.val_meter.update(outputs, labels, images.shape[0])
                        # Compute gradient and do SGD step
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                self.__write_scalars(phase, epoch)
                self.__scheduler_step(phase, epoch)
                self.__val_processing(phase, epoch)

        self.summarywriter.close()

    def __write_scalars(self, phase, epoch):
        self.summarywriter.add_scalar('Loss/{}'.format(phase),
                                      self.loss_meter.avg, epoch)
        self.__LOGGER.info('({}) loss: {}'. format(
            phase, round(self.loss_meter.avg, 8)))
        if self.val_meter.metric_name:
            self.summarywriter.add_scalar('Metric-{}/{}'.format(self.val_meter.metric_name, phase),
                                          self.val_meter.avg, epoch)
            self.__LOGGER.info('({}) metric: {}'. format(
                phase, round(self.val_meter.avg, 8)))
        if phase == 'val':
            self.summarywriter.add_scalar(
                'LR', self.optimizer.param_groups[0]['lr'], epoch)

    def __scheduler_step(self, phase, epoch):
        if phase != 'val' or self.scheduler == None:
            return
        if self.val_meter.metric_name:
            avg = self.val_meter.avg
        else:
            avg = self.loss_meter.avg
        self.scheduler.step(avg, epoch)
        if self.scheduler.num_bad_epochs == 0 and self.scheduler.best != avg:
            self.__LOGGER.info('({}) reducing LR by {} to {})'.format(
                phase, self.scheduler.factor, self.optimizer.param_groups[0]['lr']))

    def __val_processing(self, phase, epoch):
        if phase != 'val':
            return
        if self.val_meter.metric_name == None:
            avg_meter = self.loss_meter
        else:
            avg_meter = self.val_meter
        if avg_meter.update_best_avg():
            self.__LOGGER.info('({}) new best checkpoint'.format(phase))
            best_checkpoint = {'epoch': epoch, 'state_dict': self.model.state_dict(),
                               'best_avg': avg_meter.best_avg, 'optimizer': self.optimizer.state_dict()}
            utils.misc.save_checkpoint(best_checkpoint, self.output_dir)

    def test(self, data_loader: dict):
        for phase in data_loader.keys():
            self.__LOGGER.info('-' * 10)
            self.__LOGGER.info(
                'Testing on {} images'.format(len(data_loader[phase])))
            self.model.eval()
            self.val_meter.reset()
            for i, (images, labels) in enumerate(data_loader[phase]):
                # Sent images and labels to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Forward, track history if only in train
                outputs = self.model(images)
                self.val_meter.update(outputs, labels, images.shape[0])
            self.__LOGGER.info('({}) metric: {}'. format(
                phase, round(self.val_meter.avg, 8)))
        return self.val_meter.avg


if __name__ == '__main__':
    pass

"""
@history
__version__ = "1.0.0" -> basic functionality
"""
