# DEBRA
## Project Introduction
This project primarily introduces a novel on-orbit self-supervised split learning framework designed
to achieve real-time in-orbit feature extraction from satellite monitoring data. The framework mainly includes
the following components: self-supervised contrastive learning, split learning, on-orbit distributed 
training optimization algorithms, and model split optimization algorithms. Details are provided in the file structure.
## ENVS
* python>=3.8
* torch                        2.1.1+cu121
* torchaudio                   2.1.1+cu121
* torchvision                  0.16.1+cu121
* numpy                        1.26.4
* pandas                       2.2.1


## File Structure
```
DEBRA/ 
▾ checkpoints/
▾ configs/ # Configuration files
▾ data_transform/
    __init__.py
    eval_aug.py # Test set data augmentation
    simsiam_aug.py # Training set data augmentation
▾ dataset/ # Dataset path
    __init__.py
    compute_std_mean.py 
    dataset_split.py # Dataset split for test set and training set
▾ networks/ # System model
    __init__.py
    resnet.py # Backbone
    simsiam.py # Contrastive module
▾ optimize_algorithm/
    tools/ # Computing latency tools
    MWO.py # Proposed optimization algorithm
▾ optimizers/
    __init__.py
    lr_scheduler.py # Learning rate optimization based on cosine annealing
▾ split_train/ # Model split training
    __init__.py
    network/ The split model
    GA_ssl_train.py # Gradient accumulation training
    satellite_train.py 
    set_GPU_clock.py
▾ log/
▾ tools/ Self-supervised learning tools
    __init__.py
__init__.py 
arguments.py # Parameter configuration file
linear_training.py # Model fine-tuning and evaluation
README.md
ssl_training.py # Self-supervised training
```

## Datasets

* UCMerced
* AID
* EuroSAT
* Directory for storing datasets: DEBRA/dataset/

## RUN

* ssl running
  ```
  python ssl_training.py
  ```
* optimization algorithm running

You need to add satellite_positions.csv to the \DEBRA\optimize_algorithm\tools\ directory. This file contains the coordinates of LEO satellites specific to your runtime environment.
## Acknowledgment

  Parts of our implementation of SimSiam and related utilities were adapted from or inspired by [Facebook Research](https://github.com/facebookresearch/simsiam) and [Patrick Hua](https://github.com/PatrickHua/SimSiam).
  We would like to thank the original authors for making the code publicly available.

