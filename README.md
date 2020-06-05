# Semantic-Segmentation

This repository is a pytorch implementation of semantic segmentation models. It comes with a general dataloader and functions for data transforming and augmentation based on numpy and scikit-image. 
This repository aims at providing a small framework to easily train and test semantic segmentation models.

## Requirements
#### Python >= 3.7, CUDA Toolkit 10.1

#### Installation
Using [conda](https://docs.conda.io/en/latest/miniconda.html) for managing virtual environments.

    $ git clone https://github.com/RichLin56/Semantic-Segmentation.git
    $ cd path/to/Semantic-Segmentation/
    $ conda create -n pytorch_semseg python==3.7
    $ activate pytorch_semseg
    $ pip install -r requirements_windows.txt
    
## Custom dataset
The custom dataset must contain training and validation pairs of images and masks. If a testset is available it should be next to `train` and `val` folders as `test`.
#### Dataset structure
                    
        dataset           # path:  /path/to/dataset
         ├── train
         |    ├──images
         |    |   └──image_1.extension
         |    |   └──image_2.extension
         |    |   └──...
         |    ├──masks
         |    |   └──mask_1.extension
         |    |   └──mask_2.extension
         |        
         |    
         ├── val
         |    ├──images
         |    |   └──image_1.extension
         |    |   └──image_2.extension
         |    |   └──...
         |    ├──masks
         |    |   └──mask_1.extension
         |    |   └──mask_2.extension
         |        
         |  
         ├── test (Optional)
         |    ├──images
         |    |   └──image_1.extension
         |    |   └──image_2.extension
         |    |   └──...
         |    ├──masks
         |    |   └──mask_1.extension
         |    |   └──mask_2.extension
         
Images and masks are matched by their name (extensions can differ).     
#### Format of images has to be the following:
- If it comes __as image__ -> `Gray or RGB image` (all common extensions allowed, handled by PIL)
- If it comes __as numpy array__ -> make sure its dimensions are in this order `(HxWxC)`

#### Format of masks has to be the following:
- Same size as the image it belongs to
- If it comes __as image__ -> `Grayscale image (2D)` (all common extensions allowed, handled by PIL)
- If it comes __as 2D numpy array__ -> make sure its dimensions are in this order `(HxW)`
- __Classes are encoded in integer values__, starting with 0 (e.g. 5 class segmentation would have masks with integer values ranging from 0 to 4)
 
## Models
The following models are currently provided:
- [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
- [U-Net with pretrained ResNet50-backbone]
- [Fully Convolutional DenseNet](https://arxiv.org/pdf/1611.09326.pdf)

In the current state of this repo, each model can be included to this framework which fulfills the following criteria:
- __self.in_channels__ must be available as class variable and has to be set in the constructor(`__init__()`). It represents the number of channels of the images that will be fed to the model (e.g. in_channels=3 for RGB images).
- __self.out_channels__ must be available as class variable and has to be set in the constructor(`__init__()`). It represents the number of classes which have to be predicted.
- The `model returns the output of the final convolution` without any applications of non-linearities.


## Loss functions:
- __Dice Loss__
- __Cross Entropy Loss__
- __Binary Dice Loss__
- __Binary Cross Entropy Loss__

## Augmentation techniques:
Augmentations are only applied to images and masks during training phase.
- __Random Affine Transformation__
- __Random Crop__
- __Random Flip__ 
- __Random Gaussian Blur__

## Provided transformations:
Transformations are applied to images and masks(sometimes, e.g. masks dont need to be normalized) during training, validation and testing phase.
- __Normalization__ (max, min_max, max_value)
- __Resize__
- __Center Crop__

## Model evaluation
Evaluation during validation or test phase based on min(loss function) or max(metric):
- __(Binary) Dice Metric__

## Logging
- __info.log__ which logs what is happening and will be copied to your output directory after the script is done or stay in /Semantic-Segmentation/log/ if something cause the script to crash
- __Tensorboard__, files for logging with Tensorboard will be stored in the given output directory

### Logging with Tensorboard
    $ activate pytorch_semseg
    $ tensorboard --logdir path/to/output_dir/


All settings can be set via the config.jsonc file:
 

      {	
         "training":
            {
               "gpu_id": 0,
               "num_epochs": 150,
               "val_metric": "None",                  // One of: "None", "dice", "binary_dice"
               "output_dir": "/path/to/output_dir",   // mkdir if does not exist
               "test_afterwards": "True",             // Option to test after training
               "augmentation":                        // Applied to train split 
               {                                      
                  "random_flip":
                  {
                     "lr": "True",                    // 50% chance to flip left&right
                     "ud": "False"                    // 50% chance to flip up&down
                  },
                  "random_affine":
                  {
                     "ACTIVATE": "True",
                     "rotation": 5,                   // deg
                     "translation": [0.0, 0.0],       // fraction of [width, height]
                     "scale": 0.2,                    // fraction of image size
                     "shear": 0                       // deg
                  },
                  "random_crop":
                  {
                     "ACTIVATE": "False",
                     "size": 128                      // pixel			
                  },
                  "random_gaussian_blur":
                  {
                     "ACTIVATE": "False"				
                  }
               },
               "data_processing":                     // Processing applied to train and validation split 
               {
                  "resize":                           // To a quadratic shape
                  {
                     "ACTIVATE": "True",
                     "size": 256                      // pixel
                  },
                  "center_crop":                      // To a quadratic shape
                  {
                     "ACTIVATE": "False",
                     "size": 128                      // pixel			
                  },
                  "normalize":
                  {
                     "ACTIVATE": "True",
                     "mode": "255"                    // One of: "max", "min_max", "int_value"
                  }
               },
               "data": 
               {
                  "root": "/path/to/data_split/",     // Contains folders "train", "val", ("test")
                  "extension": "",                    // One of: "", ".npz", ".png", ".jpg", ...
                  "batch_size": 4,
                  "num_workers": 0
               },   
               "network":                             // Fully convolutional networks
               {
                  "name": "fcdensenet",               // One of: "fcdensenet", "unet"
                  "pretrained": "",                   // Pretrained from path/to/checkpoint.pth.tar
                  "in_channels": 1,                   // Further arguments possible
                  "out_channels": 1,                  // ...
                  "growth_rate": 16,                  // ...
                  "bottleneck_layers": 6              // ...
               },			
               "loss_function": 						
               {
                  "name": "binaryceloss",             // One of: "binaryceloss", "celoss", "binarydiceloss", "diceloss"
                  "pos_weight": 10                    // Further arguments possible
               },	
               "optimizer":  							
               {
                  "name": "adam",                     // One of: pytorch optimizers
                  "lr": 0.001                         // Further arguments possbile
               },
               "scheduler": 
               {
                  "ACTIVATE": "True",
                  "name": "reducelronplateau",        // One of: pytorch schedulers (Only "reducelronplateau" tested)
                  "factor": 0.1,                      // Further arguments possible
                  "patience": 6,                      // ...
                  "cooldown": 0                       // ...
               }

            }	
      }

The order of applied augmentation/data_processing is from top to bottom.
If you want to e.g. do a random_crop before a random_flip, just put random_crop above random_flip inside the config.jsonc. <br >

You have two ways of excluding augmentation/data_processing from your training/validation:
 1. Set the field "ACTIVATE" to "False" (recommended)
 2. Delete the not wanted augmentation/data_processing from the config.jsonc
 
This repository was developed and used (from me) during my master thesis at [LfB](https://www.lfb.rwth-aachen.de/en/).


## To do:
- [ ] TODOs in predict.py
- [ ] Update Readme
- [ ] Mean-Std Normalization
   - [ ] Implement
   - [ ] Test
- [ ] Random Salt&Pepper from skimage
   - [ ] Implement
   - [ ] Test
- [ ] Write losses:
   - [x] BinaryDice 
     - [x] Implement
     - [x] Test
   - [x] BinaryCE 
     - [x] Implement
     - [x] Test
   - [ ] Dice 
     - [x] Implement
     - [ ] Test
   - [ ] CE 
     - [x] Implement
     - [ ] Test
- [ ] Test metric functions:
  - [x] binary_dice
  - [ ] dice
- [ ] Add models:
  - [x] U-Net  
  - [x] Fully Convolutional DenseNet
  - [ ] Fully Convolutional ResNet
  - [ ] Pretrained Models
- [ ] Improve SummaryWriter with images
- [ ] Lint code


## Done:
- [x] Augmentation with is_mask flag
- [x] Saving of log file to output directory
- [x] Get Trainer.train() to run
- [x] Get Trainer.test() to run
- [x] Scheduler step based on metric evaluation (if specified)
   - [x] Write
   - [x] Test
- [x] Best checkpoint based on metric evaluation (if specified)
   - [x] Write
   - [x] Test
- [x] Add MetricMeter to Trainer.train():
  - [x] Dice metric as function 
  - [x] Implement MetricMeter in Trainer.train()
  - [x] Add config options
- [x] Move Center Crop to data_processing
- [x] Random Flip
   - [x] Implement 
   - [x] Test
