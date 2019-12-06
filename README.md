# Edge Aware Monocular Depth Estimation 
This repo contains an implementation of Resnext baseline for depth estimation in Pytorch.

## Prerequisites
Download [NYUV2 dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and [pre-trained models]().
Then follow the installation instruction below:

### Installation
1. Install Anaconda3

2. Run the following commands to create conda environment and install all dependencies:

```console
username@PC:~$ conda env create -f environment.yml
username@PC:~$ conda activate mde
```
3. You can modify cfg, train_args and test_args to change training and testing settings.


## Acknowledgments
This implementation is partially based on official implementation of [Enforcing geometric constraints of virtual normal for depth prediction.](https://github.com/YvanYin/VNL_Monocular_Depth_Prediction).
