import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import os
from typing import List, Tuple, Callable, Union, Type
from IPython.display import clear_output
import wandb

class FlexibleCNN(pl.LightningModule):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        conv_filters: List[int] = [32, 64, 128, 256, 512],
        kernel_sizes: Union[int, List[int]] = 3,
        conv_activation: Union[str, Type[nn.Module]] = "relu",
        dense_neurons: int = 512,
        dense_activation: Union[str, Type[nn.Module]] = "relu",
        pooling_size: Union[int, List[int]] = 2,
        learning_rate: float = 0.001,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0
    ):
        """
        Flexible CNN model with 5 conv-activation-maxpool blocks
        
        Args:
            input_channels: Number of input image channels (3 for RGB)
            num_classes: Number of output classes
            conv_filters: List of filter counts for each conv layer
            kernel_sizes: Kernel size for conv layers (int or list)
            conv_activation: Activation function for conv layers
            dense_neurons: Number of neurons in the dense layer
            dense_activation: Activation function for dense layer
            pooling_size: Max pooling size (int or list)
            learning_rate: Learning rate for optimizer
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (if use_dropout is True)
            use_dropout: Whether to use dropout
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Convert activation strings to functions
        self.conv_activation = self._get_activation(conv_activation)
        self.dense_activation = self._get_activation(dense_activation)
        
        # Convert single value to lists if needed
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * 5
        if isinstance(pooling_size, int):
            pooling_size = [pooling_size] * 5
            
        self.use_dropout = False if dropout_rate == 0.0 else True

        # Create 5 convolution blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = input_channels
        
        for i in range(5):
            layers = []
            
            # Convolution layer
            layers.append(nn.Conv2d(in_channels, conv_filters[i], kernel_size=kernel_sizes[i], padding=kernel_sizes[i]//2))
            
            # Optional batch normalization (before activation)
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(conv_filters[i]))
            
            # Activation
            layers.append(self._get_activation_layer(self.conv_activation))
            
            # Optional dropout after activation but before pooling
            if self.use_dropout:
                layers.append(nn.Dropout2d(dropout_rate))
            
            # Max pooling
            layers.append(nn.MaxPool2d(kernel_size=pooling_size[i], stride=pooling_size[i]))
            
            self.conv_blocks.append(nn.Sequential(*layers))
            in_channels = conv_filters[i]
        
        # Calculate the size of the flattened features
        # Assuming input image size is 224x224 (common for iNaturalist)
        # Each pooling with size 2 reduces dimensions by half
        final_size = 224 // (2 ** 5)  # After 5 pooling layers
        self.flat_size = final_size * final_size * conv_filters[-1]
        
        # Dense layer
        self.fc1 = nn.Linear(self.flat_size, dense_neurons)
        
        # Optional batch normalization for dense layer
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn_fc = nn.BatchNorm1d(dense_neurons)
            
        # Dense layer dropout (applied after activation)
        self.dropout_rate = dropout_rate
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc2 = nn.Linear(dense_neurons, num_classes)
        
        self.learning_rate = learning_rate
    
    def _get_activation(self, activation):
        """Convert activation name to function or return the provided activation"""
        if isinstance(activation, str):
            activation = activation.lower()
            if activation == 'relu':
                return F.relu
            elif activation == 'leaky_relu':
                return F.leaky_relu
            elif activation == 'elu':
                return F.elu
            elif activation == 'tanh':
                return F.tanh
            elif activation == 'sigmoid':
                return F.sigmoid
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        return activation
    
    def _get_activation_layer(self, activation_fn):
        """Convert activation function to layer"""
        if activation_fn == F.relu:
            return nn.ReLU()
        elif activation_fn == F.leaky_relu:
            return nn.LeakyReLU()
        elif activation_fn == F.elu:
            return nn.ELU()
        elif activation_fn == F.tanh:
            return nn.Tanh()
        elif activation_fn == F.sigmoid:
            return nn.Sigmoid()
        else:
            # For custom activations, use a Lambda layer
            return nn.Identity()  # Placeholder, will use functional activation in forward
    
    def forward(self, x):
        # Apply all conv blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply dense layer
        x = self.fc1(x)
        
        # Apply batch norm if enabled
        if self.use_batch_norm:
            x = self.bn_fc(x)
        
        # Apply activation
        x = self.dense_activation(x)
        
        # Apply dropout if enabled
        if self.use_dropout:
            x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class iNaturalistDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        use_data_augmentation: bool = False,
        seed=42,
    ):
        super().__init__()
        self.train_data_dir = os.path.join(data_dir, 'train')
        self.test_data_dir = os.path.join(data_dir, 'val')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.use_data_augmentation = use_data_augmentation
        self.seed = seed
        
    def setup(self, stage=None):
        # Basic transformations always applied
        basic_transforms = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        # Additional augmentations when enabled
        augmentation_transforms = []
        if self.use_data_augmentation:
            augmentation_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0))
            ]
        
        # Combine transformations
        train_transforms = transforms.Compose(augmentation_transforms + basic_transforms)
        val_transforms = transforms.Compose(basic_transforms)
        
        # Load dataset
        full_dataset = ImageFolder(root=self.train_data_dir)
        
        # Split dataset
        dataset_size = len(full_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        # Use random split with generator for reproducibility
        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )
        
        self.train_dataset.dataset = ImageFolder(root=self.train_data_dir, transform=train_transforms) 
        self.val_dataset.dataset = ImageFolder(root=self.train_data_dir, transform=val_transforms) 
        
        # # Apply transformations
        # self.train_dataset = TransformedSubset(self.train_dataset, train_transforms)
        # self.val_dataset = TransformedSubset(self.val_dataset, val_transforms)
        
        self.test_dataset = ImageFolder(root=self.test_data_dir, transform=val_transforms)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
