"""misc.py: Contains miscellaneous functions and/or classes."""

__author__ = "Richard Lindenpuetz"
__email__ = "richard.lindenpuetz@rwth-aachen.de"
__license__ = "MIT"
__version__ = "1.0.0"

import importlib
import inspect
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim


def time_now(format="%Y-%m-%d_%I-%M-%S_%p"):
    return datetime.now().strftime(format)


def find_class_in_module_by_name(module, class_name):
    class_members = [(obj, obj_name) for obj_name,
                     obj in inspect.getmembers(module) if inspect.isclass(obj)]
    class_members_name = [obj_name for obj, obj_name in class_members]
    class_members_with_class_name = [
        (obj, obj_name) for obj, obj_name in class_members if obj_name.lower() == class_name.lower()]
    assert len(class_members_with_class_name) != 0, "check config file or args!\nclass_name=\'{}\' not found in module={}\navailable class_names={}".format(
        class_name.lower(), module, class_members_name)
    assert len(class_members_with_class_name) <= 1, "multiple matches of class_name=\'{}\' found in module={}".format(
        class_name.lower(), module)
    obj = class_members_with_class_name[0][0]
    return obj


def find_module_in_package_by_name(package, module_name):
    modules = [file for file in os.listdir(package.__path__[0]) if (
        file.endswith('.py') and file != '__init__.py')]
    assert module_name.lower() + '.py' in modules, "check config file or args!\nmodule_name=\'{}\' not found in package={}\navailable modules={}".format(module_name.lower(), package, modules)
    sub_module = importlib.import_module(
        '{}.{}'.format(package.__name__, module_name.lower()))
    return sub_module


def save_checkpoint(state, dir, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dir, filename))


def one_hot_encoder(labels, num_classes):
    assert len(labels.shape) == 3
    assert labels.dtype == torch.int64
    assert num_classes > 1

    batch_size, height, width = labels.shape

    one_hot = torch.zeros(batch_size, num_classes, height, width)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
    return one_hot


def overlay_img_mask():
    pass


def tensor_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def scale_uint8(array):
    if array.max() > 0:
        array = array / array.max()
        array = np.array(array * 255, dtype=np.uint8)
    return array


def chw_to_hwc(array):
    array = np.transpose(array, axes=(1, 2, 0))
    return array



# Define a class to log values during training
class AverageMeter(object):
    def __init__(self, best_avg=1000000, comp=min):
        self.reset()
        self.best_avg = best_avg
        self.comp = comp

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def update_best_avg(self):
        avgs = [self.best_avg, self.avg]
        if avgs.index(self.comp(avgs)):
            self.best_avg = self.avg
            return True
        return False

# Define a class to log a specified metric during training
class MetricMeter(AverageMeter):
    def __init__(self, metric, best_avg=0, comp=max):
        super().__init__(best_avg=best_avg, comp=comp)
        assert metric in ['dice', 'binarydice', None]
        self.metric_name = metric
        if metric is None:
            self.metric = getattr(MetricMeter, "_MetricMeter__" + "none")
        else:
            self.metric = getattr(MetricMeter, "_MetricMeter__" + metric)
        self.reset()

    def update(self, predict, target, n):
        self.val = self.metric(predict, target)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def is_none(self):
        return self.metric == getattr(MetricMeter, "_MetricMeter__" + "none")

    def __dice(predict, target, smooth=1e-6):
        predict = torch.softmax(predict, dim=1)  # Apply softmax
        target = one_hot_encoder(labels=target.squeeze().long(
        ), num_classes=predict.shape[1])  # Apply one_hot_encoding
        # Predict shape: NxCxd1xd2xdn -> NxCxD (D=d1*d2*...dn)
        # Predict dtype: float()
        predict = predict.view(predict.shape[0], predict.shape[1], -1)
        # Target shape:  NxCxd1xd2xdn -> NxCxD (D=d1*d2*...dn)
        # Target dtype: long()
        target = target.view(target.shape[0], target.shape[1], -1)
        dims = (1, 2)
        intersection = torch.mul(predict, target)

        numerator = 2.0 * torch.sum(intersection, dim=dims) + smooth
        denominator = torch.sum(predict, dims) + \
            torch.sum(target, dims) + smooth

        dice_score = numerator / denominator
        return dice_score.mean().item()

    def __binarydice(predict, target, smooth=1e-6):
        # Predict shape: Nxd1xd2xdn -> NxD (D=d1*d2*...dn)
        # Predict dtype: float()
        predict = torch.sigmoid(predict)  # Apply sigmoid
        predict = predict.contiguous().view(predict.shape[0], -1)
        # Target shape: Nxd1xd2xdn -> NxD (D=d1*d2*...dn)
        # Target dtype: long()
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.mul(predict, target)

        numerator = 2.0 * torch.sum(intersection, dim=1) + smooth
        denominator = torch.sum(predict, dim=1) + \
            torch.sum(target, dim=1) + smooth

        dice_score = numerator / denominator
        return dice_score.mean().item()

    def __none(predict, target):
        return 0


if __name__ == '__main__':
    pass
    
"""
@history
__version__ = "1.0.0" -> basic functionality
"""
