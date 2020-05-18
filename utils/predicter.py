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

        return output.cpu().detach().numpy()    

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
        output = self.__predict(image)
        output = self.__threshold(output)

        # TODO: 
        # 1. Convert output into colored class map
        # 2. Create overlay with input
        # 3. Save into output_dir
        if output.shape[0] == 1:
            img_output = Image.fromarray(np.array(output[0]* (255 / output[0].max()), dtype=np.uint8))
        else:
            img_output = Image.fromarray(np.array(output[0]* (255 / output.max()), dtype=np.uint8))
        Image.open(image_path).save(os.path.join(self.output_dir, "org_" + os.path.basename(image_path)))
        img_output.save(os.path.join(self.output_dir, "pred_" + os.path.basename(image_path)))








if __name__ == '__main__':
    pass

"""
@history
__version__ = "1.0.0" -> basic functionality
"""