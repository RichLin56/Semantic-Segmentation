#!/usr/bin/env python

"""builder.py: Simple helper to call pytorch classes with args from .json."""

__author__ = "Richard Lindenpuetz"
__email__ = "richard.lindenpuetz@rwth-aachen.de"
__license__ = "MIT"
__version__ = "1.0.0"

import inspect

import torch

import utils.misc as misc


class Builder(object):
    def __init__(self, group_cfg, package=None, module=None):
        assert package != None or module != None
        self.name = group_cfg['name']
        if 'ACTIVATE' in group_cfg:
            if group_cfg['ACTIVATE'] == False:
                return None
        if package is not None:
            module = misc.find_module_in_package_by_name(
                package=package, module_name=self.name)
        self.__build = misc.find_class_in_module_by_name(
            module=module, class_name=self.name)


class ModelBuilder(Builder):
    def __init__(self, group_cfg, package, device):
        super().__init__(group_cfg=group_cfg, package=package)
        params_from_dict = dict(item for item in group_cfg.items(
        ) if item[0] in inspect.getargspec(self._Builder__build.__init__).args)
        self.model = self._Builder__build(**params_from_dict).to(device)
        if group_cfg['checkpoint']: 
            checkpoint = torch.load(group_cfg['checkpoint'])           
            self.model.load_state_dict(checkpoint['state_dict'])


class OptimizerBuilder(Builder):
    def __init__(self, group_cfg, package, model):
        super().__init__(group_cfg=group_cfg, package=package)
        params_from_dict = dict(item for item in group_cfg.items(
        ) if item[0] in inspect.getargspec(self._Builder__build.__init__).args)
        self.optimizer = self._Builder__build(
            model.parameters(), **params_from_dict)


class SchedulerBuilder(Builder):
    def __init__(self, group_cfg, module, optimizer, val_meter):
        super().__init__(group_cfg=group_cfg, module=module)
        params_from_dict = dict(item for item in group_cfg.items(
        ) if item[0] in inspect.getargspec(self._Builder__build.__init__).args)
        if val_meter.metric_name and group_cfg['name'] == 'reducelronplateau':
            params_from_dict['mode'] = val_meter.comp.__name__
        self.scheduler = self._Builder__build(optimizer, **params_from_dict)


class CriterionBuilder(Builder):
    def __init__(self, group_cfg, module, device):
        super().__init__(group_cfg=group_cfg, module=module)
        params_from_dict = dict(item for item in group_cfg.items(
        ) if item[0] in inspect.getargspec(self._Builder__build.__init__).args)
        if 'pos_weight' in params_from_dict:
            params_from_dict['pos_weight'] = torch.tensor(
                int(params_from_dict['pos_weight']))
        self.criterion = self._Builder__build(**params_from_dict).to(device)


if __name__ == '__main__':
    pass
    
"""
@history
__version__ = "1.0.0" -> basic functionality
"""
