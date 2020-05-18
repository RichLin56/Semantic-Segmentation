#!/usr/bin/env python

"""augmentation.py: Augmentation and transformation of images based on numpy and scikit-image."""

__author__ = "Richard Lindenpuetz"
__email__ = "richard.lindenpuetz@rwth-aachen.de"
__license__ = "MIT"
__version__ = "1.0.1"


import skimage.transform
import torch
import numpy as np
from skimage.filters import gaussian
from skimage.transform import AffineTransform, SimilarityTransform
from skimage.transform import PolynomialTransform, warp


def deterministic_affine(rotation=0, translation=[0.0, 0.0], scale=1.0, shear=0):
    def call(array):
        assert isinstance(rotation, int)
        assert isinstance(translation, list) and len(translation) == 2
        assert isinstance(scale, float)
        assert isinstance(shear, int)
        assert scale >= 0
        assert array.ndim <= 3, "array.ndim > 3 is not supported"
        assert all([isinstance(translation[i], float) for i in range(2)])
        # Prepare Affine Transformation
        width = array.shape[-1]
        height = array.shape[-2]
        center_shift = (width/2, height/2)
        tf_center = SimilarityTransform(translation=np.negative(center_shift))
        tf_uncenter = SimilarityTransform(translation=center_shift)

        rotation_tmp = rotation
        translation_tmp = np.array(translation) * np.array((-width, height))
        scale_tmp = (scale, scale)
        shear_tmp = shear

        tf_augment = AffineTransform(
            scale=scale_tmp,
            rotation=np.deg2rad(rotation_tmp),
            translation=translation_tmp,
            shear=np.deg2rad(shear_tmp)
        )
        tf = tf_center + tf_augment + tf_uncenter
        # Affine Transformation
        if array.ndim == 2:
            array = warp(array, tf, order=0,
                         preserve_range=True, mode='constant')
        if array.ndim == 3:
            array = np.array([warp(array[i], tf, order=0, preserve_range=True,
                                   mode='constant') for i in range(len(array))])
        return array

    return call


def random_affine(rotation=0, translation=[0.0, 0.0], scale=1.0, shear=0):
    def call(array, is_mask=False):
        assert isinstance(rotation, int)
        assert isinstance(translation, list)
        assert isinstance(scale, float)
        assert isinstance(shear, int)
        assert rotation >= 0
        assert scale > 0 and scale <= 1.0, " 0 < scale <= 1.0"
        assert shear >= 0, "shear < 0 not allowed"
        assert len(translation) == 2
        assert all([translation[i] >= 0.0 for i in range(2)]
                   ), "translation values >= 0.0"
        assert all([isinstance(translation[i], float)
                    for i in range(2)]), "translation values are not of type float"
        assert array.ndim <= 3, "array.ndim > 3 is not supported"
        # Randomize values

        def randomize_value_around_zero(min_max_value):
            rnd_value = np.random.randint(-min_max_value *
                                          100, min_max_value * 100) / 100
            return rnd_value
        # Random rotation
        if rotation != 0:
            rotation_tmp = randomize_value_around_zero(min_max_value=rotation)
        else:
            rotation_tmp = 0
        # Random translation
        if translation[0] != 0.0:
            translation_tmp_x = randomize_value_around_zero(
                min_max_value=translation[0])
        else:
            translation_tmp_x = 0.0
        if translation[1] != 0.0:
            translation_tmp_y = randomize_value_around_zero(
                min_max_value=translation[1])
        else:
            translation_tmp_y = 0.0
        # Random scale
        if scale != 1.0:
            scale_tmp = randomize_value_around_zero(min_max_value=scale) + 1.0
        else:
            scale_tmp = 1.0
        # Random shear
        if shear != 0:
            shear_tmp = randomize_value_around_zero(min_max_value=shear)
        else:
            shear_tmp = 0

        # Prepare Affine Transformation
        width = array.shape[-1]
        height = array.shape[-2]
        center_shift = (width/2, height/2)
        tf_center = SimilarityTransform(translation=np.negative(center_shift))
        tf_uncenter = SimilarityTransform(translation=center_shift)

        translation_tmp = np.array(
            (translation_tmp_x, translation_tmp_y)) * np.array((-width, height))
        scale_tmp = (scale_tmp, scale_tmp)

        tf_augment = AffineTransform(
            scale=scale_tmp,
            rotation=np.deg2rad(rotation_tmp),
            translation=translation_tmp,
            shear=np.deg2rad(shear_tmp)
        )
        tf = tf_center + tf_augment + tf_uncenter
        # NN-interpolation for mask
        order = 1
        if is_mask:
            order = 0
        # Affine Transformation
        if array.ndim == 2:
            array = warp(array, tf, order=order,
                         preserve_range=True, mode='constant')
        if array.ndim == 3:
            array = np.array([warp(array[i], tf, order=order, preserve_range=True,
                                   mode='constant') for i in range(len(array))])
        return array

    return call


def random_crop(size):
    def call(array, is_mask=False):
        assert isinstance(size, int), "size is not of type int"
        width = array.shape[-1]
        height = array.shape[-2]
        assert size <= min(
            width, height), "size > min(width, height) of array not allowed"
        assert array.ndim <= 3, "array.ndim > 3 is not supported"
        # Randomize values
        r1 = np.random.random()
        r2 = np.random.random()

        left = round(r1 * (width - size))
        right = round((1 - r1) * (width - size))

        top = round(r2 * (height - size))
        bottom = round((1 - r2) * (height - size))
        # Random crop
        if array.ndim == 2:
            crop = array[top:height-bottom, left:width-right]
        if array.ndim == 3:
            crop = np.array(
                [array[i, top:height-bottom, left:width-right] for i in range(len(array))])
        return crop

    return call


def center_crop(size):
    def call(array, is_mask=False):
        assert isinstance(size, int), "size is not of type int"
        width = array.shape[-1]
        height = array.shape[-2]
        assert size <= min(
            width, height), "size > min(width, height) of array not allowed"
        assert array.ndim <= 3, "array.ndim > 3 is not supported"
        # Center values
        a = int(0.5 * (height - size))
        b = int(0.5 * (width - size))
        top, bottom = a, (height - size) - a
        left, right = b, (height - size) - b
        # Crop
        if array.ndim == 2:
            crop = array[top:height-bottom, left:width-right]
        if array.ndim == 3:
            crop = np.array(
                [array[i, top:height-bottom, left:width-right] for i in range(len(array))])
        return crop

    return call


def resize(size):
    def call(array, is_mask=False):
        assert isinstance(size, int), "size is not of type int"
        assert size > 0, "size <= 0 not allowed"
        assert array.ndim <= 3, "array.ndim > 3 is not supported"
        # Determine anti aliasing flag
        width = array.shape[-1]
        height = array.shape[-2]
        aa = min(width, height) > size
        order = 1
        if is_mask:
            order = 0
            aa = False
        # Resize array
        if array.ndim == 2:
            resized = skimage.transform.resize(array, (size, size), preserve_range=True,
                                               order=order, anti_aliasing=aa)
        if array.ndim == 3:
            resized = np.array(
                [skimage.transform.resize(array[i], (size, size), preserve_range=True,
                                          order=order, anti_aliasing=aa) for i in range(len(array))])
        return resized

    return call


def deterministic_gaussian_blur(sigma):
    def call(array):
        assert isinstance(sigma, float), "size is not of type float"
        assert sigma >= 0, "size < 0 is not allowed"
        assert array.ndim <= 3, "array.ndim > 3 is not supported"
        # Gaussian blur
        array = gaussian(array, sigma=sigma,
                         preserve_range=True, multichannel=True)
        return array
    return call


def random_gaussian_blur():
    def call(array, is_mask=False):
        assert array.ndim <= 3, "array.ndim > 3 is not supported"
        # Random value
        sigma = np.random.random_sample()
        # No blur on mask
        if is_mask:
            return array
        # Gaussian blur
        array = gaussian(array, sigma=sigma,
                         preserve_range=True, multichannel=True)
        return array
    return call


def normalize(mode='min_max'):
    def call(array, is_mask=False):
        assert mode in [
            'min_max', 'max'] or isinstance(int(mode), int)
        # No normalization on mask
        if is_mask:
            return array
        if mode == 'min_max':
            return normalize_min_max()(array)
        if mode == 'max':
            return normalize_max()(array)
        if isinstance(int(mode), int):
            return normalize_by_value()(array, value=int(mode))
    return call


def normalize_max():
    def call(array):
        max_val = array.max()
        if max_val > 0:
            array = array / max_val
        return array

    return call


def normalize_by_value():
    def call(array, value):
        assert value != 0
        array = array / value
        return array

    return call


def normalize_min_max():
    def call(array):
        min_val = array.min()
        max_val = array.max()
        if max_val != min_val:
            array = (array - min_val) / (max_val - min_val)
        return array

    return call


def random_flip(lr=False, ud=False):
    def call(array, is_mask=False):
        assert array.ndim <= 3, "array.ndim > 3 is not supported"
        prob_lr = np.random.randint(2)
        prob_ud = np.random.randint(2)
        if lr == True and prob_lr:
            if array.ndim == 2:
                array = np.fliplr(array)
            if array.ndim == 3:
                array = np.array(
                        [np.fliplr(array[i]) for i in range(len(array))])
        if ud == True and prob_ud: 
            if array.ndim == 2:
                array = np.flipud(array)
            if array.ndim == 3:
                array = np.array(
                        [np.flipud(array[i]) for i in range(len(array))])
        return array
    
    return call
            

def to_float_64():
    def call(array, is_mask=False):
        float64 = np.asarray(array, dtype=np.float64)
        return float64

    return call


def to_float_32():
    def call(array, is_mask=False):
        float32 = np.asarray(array, dtype=np.float32)
        return float32

    return call


def to_tensor():
    def call(array, is_mask=False):
        tensor = torch.from_numpy(array)
        return tensor

    return call


def flatten_list(flat_me, flat_into):
    for i in flat_me:
        if isinstance(i, list):
            flatten_list(i, flat_into)
        else:
            flat_into.append(i)
    return flat_into


def apply_chain(trafos_data_processing, trafos_augmentation=None):
    def call(array, is_mask=False):
        trafos_tmp = flatten_list([
            trafos_augmentation,
            trafos_data_processing,
            to_float_32(),
            to_tensor()
        ], [])
        for trafo in trafos_tmp:
            if trafo is None:
                continue
            array = trafo(array, is_mask=is_mask)
        return array

    return call


if __name__ == '__main__':
    pass
    
"""
@history
__version__ = "1.0.0" -> basic functionality
"""
