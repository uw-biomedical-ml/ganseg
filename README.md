# GANSeg
[Training deep learning models to work on multiple devices by cross domain learning with no additional annotations](https://www.aaojournal.org/article/S0161-6420%2822%2900749-7/fulltext)

This repository contains the PyTorch implementation of GANSeg. GANSeg consists of two networks: a) a GAN and b) a supervised network. 
For this implementation of GANSeg, we focused on applying GANSeg to cross domain segmentation. 
Therefore the supervised network used was a [U-Net](https://arxiv.org/abs/1505.04597).
The GAN network that was used is adapted from [U-GAT-IT](https://arxiv.org/abs/1907.10830, https://github.com/znxlwm/UGATIT-pytorch). 

In our framework the GAN network is used to generate images in the target domain (different OCT machine) while retaining the pixel location of the retinal anatomy and intraretinal fluid. Thus the U-Net is supervised in the source domain where annotations are present and unsupervised in the target domain. 

# DataSets
# Source Dataset
The original dataset for cross domain adaptation was the Duke SD-OCT dataset for DME patients. 
This dataset Heidelberg Spectralis SD-OCT volumes from 10 patients suffering from DME. 
Each Heidelberg Spectralis volume consist of 61 scans.
Of these, 11 B-scans were annotated for each patients resulting in 110 total annotated images of size 512 × 740. 
The 11 B-scans per patient were annotated centered at fovea and 5 frames on either side of the fovea (foveal slice and scans laterally acquired at ±2, ±5, ±10, ±15 and ± 20 from the foveal slice). 
These 110 B-scans are annotated for the retinal layers and fluid regions by two expert clinicians. The details of this datasets in this [home-page](http://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm).

# Regraded Dataset
We noticed substantial variability in the grading of the intra-retinal fluid.
Therefore, our clinical experts at Moorfields Eye Hospital regraded the intra-retinal fluid of all 110 annotated B-scans.
The images and masks in the ./data directory are the regraded intra-retinal fluid overlaid on the original Duke layer annotations.

# Target Dataset
For our experiments, we extracted raw (un-annotated) OCT B-scans from UKB iobank. 
The OCT B-scans in the UK Biobank were taken with Topcon 1000. 
We have not included the UK Biobank B-scans in this repo, since that dataset is already available to researchers upon application.

# Other Datasets
We also included 3 Zeiss Plex Elite B-scans and the corresponding annotations by 2 Moorfields graders.
Finally, we included 3 Topcon Maestro2 B-scans and the corresponding annotations by 2 Moorfields graders.

# Code
# Training
To run the code for training,
```python main.py --phase train --device cuda --dataset [your_dataset] --batch_size 16 --img_ch 1 --seg_classes 9 --seg_visual_factor 30 --adv_weight 1 --cycle_weight 10 --identity_weight 10 --cam_weight 1000 --seg_weight 1000 --aug_options_file aug_options.json --add_seg_link 1  --no_gan 1 --result_dir [your_result_dir] --iteration 3000```

# Prediction
```python main.py --phase test --device cuda --dataset [your_dataset] --batch_size 16 --img_ch 1 --seg_classes 9 --seg_visual_factor 30 --adv_weight 1 --cycle_weight 10 --identity_weight 10 --cam_weight 1000 --seg_weight 1000 --aug_options_file aug_options.json --add_seg_link 1  --no_gan 1 --result_dir [your_result_dir] --iteration 3000 --testB_folder [testB_image_folder]```

# Trained Weights
Trained weights for GANSeg to segment 7 retinal layers and intra-retinal fluid for both Heidelberg Spectralis and Topcon 1000 Bscans are provided [here](https://drive.google.com/file/d/1XVw778KiKc6YTNq6esxat-jEQlA5M5uV/view?usp=sharing).
