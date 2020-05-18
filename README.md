# Semantic-Segmentation
This repository is a pytorch implementation of semantic segmentation models. It comes with a general dataloader and functions for data transforming and augmentation based on numpy and scikit-image. 
This repository aims at providing a small framework to train and test semantic segmentation models.

The following models are currently provided:
- [U-Net](https://arxiv.org/pdf/1505.04597.pdf)
- [Fully Convolutional DenseNet](https://arxiv.org/pdf/1611.09326.pdf)

With the following loss functions:
- Dice Loss 
- Cross Entropy Loss 
- Binary Dice Loss
- Binary Cross Entropy Loss

Provided augmentation techniques:
- Random Affine Transformation
- Random Crop
- Random Flip 
- Random Gaussian Blur

Provided transformations:
- Normalization (max, min_max, max_value)
- Resize
- Center Crop

Model evaluation based on min(loss function) or max(metric):
- Dice Metric

Provided logging:
- info.log which will be copied to your output directory after the script is done
- Tensorboard, with log file in output directory


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
