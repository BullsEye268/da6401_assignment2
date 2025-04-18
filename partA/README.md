# CNN Implementation for iNaturalist Classification

**Author**:

**WandB Report Link**:

## Overview

This project implements a Convolutional Neural Network (CNN) model for image classification on the iNaturalist dataset using PyTorch Lightning. The model architecture consists of 5 convolution layers, each followed by an activation function and a max-pooling layer, concluding with a dense layer and an output layer with 10 neurons for class prediction.

## Model Architecture

The implemented CNN model (`FlexibleCNN`) features:

- 5 convolutional blocks, each containing:
  - Convolutional layer
  - Optional batch normalization
  - Activation function
  - Optional dropout
  - Max pooling
- A flexible dense layer with configurable number of neurons
- Output layer with 10 neurons (one for each class)

The model is designed with extensive configurability, allowing for experimentation with:

- Number of filters in each layer
- Kernel sizes
- Activation functions (ReLU, GELU, SiLU, ELU, etc.)
- Filter organization patterns (constant, doubling, halving, diamond)
- Batch normalization
- Dropout rates and placement

## Data Handling

The `iNaturalistDataModule` class manages:

- Dataset loading from the iNaturalist dataset
- 80/20 train/validation split of the training data
- Optional data augmentation (horizontal flips, rotations, resized crops)
- Image resizing and normalization
- Efficient data loading with DataLoaders

## Hyperparameter Tuning

Hyperparameter optimization was conducted using Weights & Biases (wandb) sweeps with Bayesian optimization. The hyperparameters explored included:

1. **Learning rate**: Log-uniform distribution from 1e-4 to 1e-2
2. **Batch size**: 32, 64, 128
3. **Convolutional filter patterns**:
   - Standard doubling (32,64,128,256,512)
   - Constant filters (64,64,64,64,64)
   - Decreasing filters (128,96,64,48,32)
   - Diamond pattern (32,64,128,64,32)
4. **Activation functions**: ReLU, Leaky ReLU, ELU
5. **Dense layer neurons**: 128, 256, 512
6. **Batch normalization**: Enabled/Disabled
7. **Dropout rate**: 0.0, 0.2, 0.3, 0.5
8. **Data augmentation**: Enabled/Disabled

The sweep was configured to maximize validation accuracy across multiple runs, with each run training for 10 epochs.

## Implementation Details

- **Framework**: PyTorch Lightning for clean, organized training code
- **Optimizer**: Adam optimizer with configurable learning rate
- **Loss Function**: Cross-entropy loss for classification
- **Metrics**: Accuracy and loss tracked for training, validation, and testing
- **Early Stopping**: Used to prevent overfitting
- **Checkpoint Saving**: Best models saved based on validation loss

## Project Structure

- `utils.py`: Contains the model architecture and data module implementations
- `trials.ipynb`: Jupyter notebook for experimentation, training, and hyperparameter sweeps
- `data/`: Directory containing the iNaturalist dataset
- `wandb/`: Directory containing Weights & Biases logs
- `inaturalist-cnn/`, `cnn-pytorch-lightning/`: Directories for different experiment runs

## Results

The hyperparameter sweep identified optimal configurations that balance model complexity and performance. The best performing models achieved validation accuracy improvements over the baseline, with specific performance details available in the linked WandB report.
