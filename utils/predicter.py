#!/usr/bin/env python

"""predicter.py: Class to manage a prediction routine for semantic segmentation."""

__author__ = "Richard Lindenpuetz"
__email__ = "richard.lindenpuetz@rwth-aachen.de"
__license__ = "MIT"
__version__ = "1.0.0"

import os
from PIL import Image

import numpy as np
import torch

import utils.misc as misc
import utils.augmentation as augmentation


class Predicter(object):
    def __init__(self, model, output_dir, data_processing, device, threshold=0.5):
        self.model = model.model   
        self.output_dir = output_dir
        self.data_processing = augmentation.apply_chain(trafos_data_processing=data_processing)
        self.device = device    
        self.threshold = threshold 

    def __predict(self, image):
        image = self.data_processing(image, is_mask=False)
        image = image.to(self.device)
        assert image.ndim == 2 or image.ndim == 3, "image.ndim > 3 is not supported"
        if image.ndim == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        else :
            image = image.unsqueeze(0)
        
        output = self.model(image)    

        if self.model.out_channels == 1:
            output = torch.sigmoid(output)  
        else:
            output = torch.softmax(output)    
        output = output.squeeze(0)    

        return misc.tensor_to_numpy(output), misc.tensor_to_numpy(image)    

    def __threshold(self, image):
        channels = image.shape[0]
        for i in range(channels):
            temp = image[i]
            temp[temp > self.threshold] = 1.0
            temp[temp <= self.threshold] = 0.0
            image[i] = temp
        return image

    def predict(self, image_path):
        image = np.array(Image.open(image_path))
        outputs, inputs = self.__predict(image)
        outputs = self.__threshold(outputs)

        img_input = misc.scale_uint8(inputs[0])
        img_input = misc.chw_to_hwc(img_input)        
        if img_input.shape[-1] == 1:
            img_input = Image.fromarray(img_input[:, :, 0])
        else:
            img_input = Image.fromarray(img_input)

        img_output = np.array([misc.scale_uint8(outputs[i]) for i in range(outputs.shape[0])])
        img_output = misc.chw_to_hwc(img_output)
        if img_output.shape[-1] == 1:
            img_output = Image.fromarray(img_output[:, :, 0])
        else:
            # TODO: Does this case occur? If so: Implement this
            pass
        img_output.save(os.path.join(self.output_dir, "pred_" + os.path.basename(image_path)))
        img_input.save(os.path.join(self.output_dir, "org_" + os.path.basename(image_path)))



if __name__ == '__main__':
    pass

"""
@history
__version__ = "1.0.0" -> basic functionality
"""