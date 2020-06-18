#!/usr/bin/env python

"""loss_functions.py: Contains loss_functions used in semantic segmentation."""

__author__ = "Richard Lindenpuetz"
__email__ = "richard.lindenpuetz@rwth-aachen.de"
__license__ = "MIT"
__version__ = "1.0.0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class BinaryCELoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(BinaryCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(weight=weight, size_average=size_average,
                                             reduce=reduce, reduction=reduction,
                                             pos_weight=pos_weight)

    def forward(self, predict, target):
        # Predict shape: Nxd1xd2xdn -> NxD (D=d1*d2*...dn)
        # Predict dtype: float()
        predict = predict.contiguous().view(predict.shape[0], -1)
        # Target shape: Nxd1xd2xdn -> NxD (D=d1*d2*...dn)
        # Target dtype: float()
        target = target.contiguous().view(target.shape[0], -1)
        return self.bce_loss(predict, target)


class CELoss(nn.Module):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduction='mean'):
        super(CELoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index,
                                           reduction=reduction)

    def forward(self, predict, target):
        # Predict shape: NxCxd1xd2xdn -> NxCxD (D=d1*d2*...dn)
        # Predict dtype: float()
        predict = predict.contiguous().view(
            predict.shape[0], predict.shape[1], -1)
        # Target shape: Nxd1xd2xdn -> NxD (D=d1*d2*...dn)
        # Target dtype: long()
        target = target.view(target.shape[0], target.shape[1], -1)
        target = target.contiguous().view(target.shape[0], -1)
        return self.ce_loss(predict, target.long())


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, predict, target):
        # Predict shape: Nxd1xd2xdn -> NxD (D=d1*d2*...dn)
        # Predict dtype: float()
        predict = torch.sigmoid(predict)  # Apply sigmoid
        predict = predict.contiguous().view(predict.shape[0], -1)
        # Target shape: Nxd1xd2xdn -> NxD (D=d1*d2*...dn)
        # Target dtype: long()
        target = target.contiguous().view(target.shape[0], -1)

        intersection = torch.mul(predict, target)

        numerator = 2.0 * torch.sum(intersection, dim=1) + self.smooth
        denominator = torch.sum(predict, dim=1) + \
            torch.sum(target, dim=1) + self.smooth

        loss = 1 - numerator / denominator

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == None:
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, predict, target):
        predict = torch.softmax(predict, dim=1)  # Apply softmax
        target = utils.misc.one_hot_encoder(labels=target.squeeze(
        ).long(), num_classes=predict.shape[1])  # Apply one_hot_encoding
        # Predict shape: NxCxd1xd2xdn -> NxCxD (D=d1*d2*...dn)
        # Predict dtype: float()
        predict = predict.contiguous().view(
            predict.shape[0], predict.shape[1], -1)
        # Target shape:  NxCxd1xd2xdn -> NxCxD (D=d1*d2*...dn)
        # Target dtype: long()
        target = target.contiguous().view(target.shape[0], target.shape[1], -1)
        dims = (1, 2)
        intersection = torch.mul(predict, target)

        numerator = 2.0 * torch.sum(intersection, dim=dims) + self.smooth
        denominator = torch.sum(predict, dims) + \
            torch.sum(target, dims) + self.smooth

        loss = 1 - numerator / denominator

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == None:
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

# TODO: Rewrite FocalLoss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, predict, target):
        num_class = predict.shape[1]
        assert num_class > 1, "out_channels < 2 not allowed with FocalLoss"
        if predict.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            predict = predict.view(predict.size(0), predict.size(1), -1)
            predict = predict.permute(0, 2, 1).contiguous()
            predict = predict.view(-1, predict.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != predict.device:
            alpha = alpha.to(predict.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != predict.device:
            one_hot_key = one_hot_key.to(predict.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth/(num_class-1), 1.0 - self.smooth)
        pt = (one_hot_key * predict).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


if __name__ == '__main__':
    pass
    
"""
@history
__version__ = "1.0.0" -> basic functionality
"""
