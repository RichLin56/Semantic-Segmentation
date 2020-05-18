#!/usr/bin/env python

"""dataset.py: Implementation of a general dataset for semantic segmentation derived from pytorch's Dataset."""

__author__ = "Richard Lindenpuetz"
__email__ = "richard.lindenpuetz@rwth-aachen.de"
__license__ = "MIT"
__version__ = "1.0.0"

import os

import numpy as np
import skimage.io as io
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import utils.augmentation as augmentation


class SemSegDataset(Dataset):
    """
        Dataset that contains images and corresponding masks.
    """

    def __init__(self, root, subset="train", transform=None, extension_image='', extension_mask=''):
        """
        :param root: Path to the folder that contains dataset split (train, val, test)
        :param subset: Current subset (train, val, test)
        :param transform: Transformation for images and masks
        :param extension: (Optional) file format to look for in subset folder
        """

        # Initialize variables
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.subset = subset
        self.image_paths, self.mask_paths = [], []
        self.extension_image = extension_image.lower()
        self.extension_mask = extension_mask.lower()

        assert self.subset in [
            'train', 'val', 'test'], "invalid parameter subset={}, it must be one of: \'train\', \'val\', \'test\'".format(self.subset)

        def load_file_paths(directory, extension):
            """
            Load all files with specified extenstion in given directory.
            :param directory:
            :return: List with all the paths to the arrays
            """
            arrays_dir = [os.path.join(directory, f) for f in os.listdir(directory) if (
                os.path.isfile(os.path.join(directory, f)) 
                and f.endswith(extension) 
                and not f.startswith('.'))]
            arrays_dir.sort()
            return arrays_dir

        # Load the paths regarding the subset
        # Images
        image_dir = os.path.join(self.root, self.subset, 'images')
        assert os.path.isdir(image_dir), "{} does not exist".format(image_dir)
        self.image_paths = load_file_paths(image_dir, self.extension_image)
        # Masks
        mask_dir = os.path.join(self.root, self.subset, 'masks')
        assert os.path.isdir(mask_dir), "{} does not exist".format(mask_dir)
        self.mask_paths = load_file_paths(mask_dir, self.extension_mask)
        assert len(self.image_paths) == len(self.mask_paths), "mismatch between number of images {} and masks {} in subset {}".format(
            len(self.image_paths), len(self.mask_paths), self.subset)

    def __getitem__(self, index):
        """
        :param index:
        :return: tuple (image, mask) as numpy arrays with (CxHxW)
        """
        def load_image_from_file(path, extension):
            # Load numpy as array (HxWxC)
            if extension == '.npy':
                image = np.load(path)
            # Load file as array (HxWxC)
            else:
                image = io.imread(path)
            assert image.ndim <= 3, "array.ndim > 3 is not supported"
            if image.ndim == 2:
                # Rearrange to (HxW)
                image = image.transpose(1, 0)
                # Expand to (CxHxW)
                image = np.expand_dims(image, axis = 0)  
                return image        
            if image.ndim == 3:
                # Rearrange to (CxHxW)
                image = image.transpose(2, 0, 1)                        
                return image

        # Load images and masks
        if (len(self.mask_paths) > 0):
            # Load file as array
            image = load_image_from_file(self.image_paths[index], self.extension_image)
            mask = load_image_from_file(self.mask_paths[index], self.extension_mask)
            # Apply transforms
            if self.transform is not None:
                # Generate seed for transformation
                seed = np.random.randint(pow(2, 31))
                np.random.seed(seed)
                image = self.transform(image, is_mask=False)
                np.random.seed(seed)
                mask = self.transform(mask, is_mask=True)
            return image, mask
        # Only images available, e.g. sometimes when subset == 'test'
        else:            
            image = load_image_from_file(self.image_paths[index], self.extension_image)
            if self.transform is not None:
                # Generate seed for transformation
                seed = np.random.randint(pow(2, 31))
                np.random.seed(seed)
                image = self.transform(image, is_mask=False)
            return image

    def __len__(self):
        return len(self.image_paths)


class SemSegDataLoader(DataLoader):
    def __init__(self, root, subset, data_processing, augmentations=None, 
                 extension_image='', extension_mask='', batch_size=1, num_workers=0, 
                 sampler=None, batch_sampler=None, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):   
        shuffle = not(subset == 'test')
        transform = augmentation.apply_chain(trafos_data_processing=data_processing, 
                                             trafos_augmentation=augmentations)
        self.dataset = SemSegDataset(root, subset, transform, extension_image, extension_mask)
        super().__init__(self.dataset, batch_size, shuffle=shuffle, num_workers=num_workers, 
                    sampler=sampler, batch_sampler=batch_sampler, collate_fn=collate_fn,
                    pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
                    worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context)



if __name__ == '__main__':
	pass
    
"""
@history
__version__ = "1.0.0" -> basic functionality
"""
