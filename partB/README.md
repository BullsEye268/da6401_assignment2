# Fine-tuning a Pre-trained Model on iNaturalist Dataset

**Author**: Achyutha Munimakula PH21B004

[WandB Report Link](https://api.wandb.ai/links/bullseye2608-indian-institute-of-technology-madras/jxhqe65y)

This folder contains the implementation for fine-tuning a pre-trained EfficientNetV2 model on the iNaturalist_12k dataset.

## Overview

The project demonstrates transfer learning by adapting an ImageNet pre-trained model to classify images from the iNaturalist dataset with 10 classes. The implementation explores different fine-tuning strategies and compares their performance to training from scratch.

## Files

- `utils.py`: Contains utility functions for data processing, model training, and evaluation
- `trials.ipynb`: Jupyter notebook with the implementation and experiments
- `prediction_grid.png`: Visualization of model predictions

## Key Results

- Achieved 87.6% accuracy using partial fine-tuning compared to 39.6% when training from scratch
- Fine-tuning the last third of EfficientNetV2 reaches 80% validation accuracy by epoch 5
- Freezing early layers reduces compute time by approximately 40%

## Implementation Details

### Data Preparation

- Images are resized to match EfficientNetV2's expected input dimensions
- Same normalization parameters as ImageNet training are applied
- Data augmentation techniques improve generalization

### Model Adaptation

- The ImageNet pre-trained model is loaded without the final classification layer
- The final fully-connected layer is replaced with a new one having 10 output nodes for the target classes

### Fine-tuning Strategies

1. **Freeze All Layers Except the Classifier**: Only the final classification layer is trained
2. **Freeze Early Layers, Fine-Tune Later Layers**: First two-thirds of layers frozen, remaining layers fine-tuned
3. **Gradual Unfreezing**: Starting with only classifier training, then progressively unfreezing additional layers

The experiments demonstrate that reusing ImageNet-learned features with targeted fine-tuning delivers superior generalization and resource efficiency compared to end-to-end training from scratch.
