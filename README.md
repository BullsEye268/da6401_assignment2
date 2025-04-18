# DA6401 Deep Learning Assignment 2

**Author**: Achyutha Munimakula PH21B004

[WandB Report Link](https://api.wandb.ai/links/bullseye2608-indian-institute-of-technology-madras/jxhqe65y)

## Project Overview

This repository contains the implementation of Assignment 2 for the DA6401 Deep Learning course, focusing on Convolutional Neural Networks (CNNs) and transfer learning techniques.

## Part A: Building a CNN from Scratch

Located in the [partA](./partA) directory, this section implements a custom CNN architecture trained from scratch on the iNaturalist dataset. Key components include:

- Data preprocessing and augmentation pipeline
- Custom CNN architecture design and implementation
- Training and validation using PyTorch
- Hyperparameter tuning experiments
- Performance visualization and analysis

The model achieves competitive accuracy on the iNaturalist dataset while exploring the fundamentals of CNN architectures.

## Part B: Fine-tuning a Pre-trained Model

Located in the [partB](./partB) directory, this section explores transfer learning by fine-tuning a pre-trained EfficientNetV2 model on the same iNaturalist dataset. Key features include:

- Adaptation of pre-trained ImageNet weights to the iNaturalist classification task
- Implementation of different fine-tuning strategies:
  - Freezing all layers except the classifier
  - Freezing early layers and fine-tuning later layers
  - Gradual unfreezing approach
- Comparative analysis between training from scratch vs. transfer learning

The fine-tuned model demonstrates significant improvements in accuracy (87.6% vs. 39.6%) and training efficiency compared to the from-scratch approach.

## Technologies Used

- PyTorch
- PyTorch Lightning
- Wandb for experiment tracking
- Matplotlib/Seaborn for visualization

## Directory Structure

```
da6401_assignment2/
├── partA/                  # Building a CNN from scratch
│   ├── data/               # Dataset directory
│   ├── utils.py            # Utility functions
│   ├── trials.ipynb        # Implementation notebook
│   └── README.md           # Documentation
│
├── partB/                  # Fine-tuning pre-trained models
│   ├── utils.py            # Utility functions
│   ├── trials.ipynb        # Implementation notebook
│   └── README.md           # Documentation
│
└── README.md               # Main repository documentation
```
