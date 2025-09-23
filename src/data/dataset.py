import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional, List
import random

# Global class names for consistency
CLASS_NAMES = ['good', 'minor', 'moderate', 'severe']

class PotholeImageDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", image_size: int = 224, augment: bool = True):
        """
        Dataset for pothole severity classification with good road detection
        
        Args:
            data_dir: Root directory containing the dataset
            split: "train", "val", or "test"
            image_size: Target image size
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.augment = augment
        
        # Class mapping: 0=good, 1=minor, 2=moderate, 3=severe
        self.class_names = CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load data paths and labels
        self.samples = []
        self._load_data()
        
        # Set up transforms
        self.transform = self._get_transforms()
        
    def _load_data(self):
        """Load image paths and labels from the dataset structure"""
        # Load pothole images (minor, moderate, severe) from the organized structure
        pothole_dir = os.path.join(self.data_dir, 'data', 'potholes')
        if os.path.exists(pothole_dir):
            for severity in ['minor', 'moderate', 'severe']:
                severity_dir = os.path.join(pothole_dir, self.split, severity)
                if os.path.exists(severity_dir):
                    for img_name in os.listdir(severity_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                            img_path = os.path.join(severity_dir, img_name)
                            self.samples.append((img_path, self.class_to_idx[severity]))
        
        # Load plain road images (good class) from plain_image_data
        plain_dir = os.path.join(self.data_dir, 'plain_image _data')
        if os.path.exists(plain_dir):
            split_dir = os.path.join(plain_dir, self.split)
            if os.path.exists(split_dir):
                for img_name in os.listdir(split_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        img_path = os.path.join(split_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx['good']))
        
        # Also load from pothole_image_data for additional training data
        pothole_flat_dir = os.path.join(self.data_dir, 'pothole_image_data')
        if os.path.exists(pothole_flat_dir):
            split_dir = os.path.join(pothole_flat_dir, self.split)
            if os.path.exists(split_dir):
                for img_name in os.listdir(split_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        img_path = os.path.join(split_dir, img_name)
                        # Determine severity from filename or use moderate as default
                        severity = self._extract_severity_from_filename(img_name)
                        if severity in self.class_to_idx:
                            self.samples.append((img_path, self.class_to_idx[severity]))
        
        print(f"Loaded {len(self.samples)} samples for {self.split} split")
        # Print class distribution
        class_counts = {}
        for _, label in self.samples:
            class_name = self.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print(f"Class distribution: {class_counts}")
    
    def _extract_severity_from_filename(self, filename: str) -> str:
        """Extract severity from filename based on keywords"""
        filename_lower = filename.lower()
        
        if any(word in filename_lower for word in ['severe', 'critical', 'dangerous', 'deep']):
            return 'severe'
        elif any(word in filename_lower for word in ['moderate', 'medium', 'mod']):
            return 'moderate'
        elif any(word in filename_lower for word in ['minor', 'small', 'light']):
            return 'minor'
        else:
            # Default to moderate if no clear indication
            return 'moderate'
    
    def _get_transforms(self) -> transforms.Compose:
        """Get transforms based on split and augmentation setting"""
        if self.split == "train" and self.augment:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        try:
            # Load and convert image to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Ensure tensor is float32 for MPS compatibility
            if image.dtype != torch.float32:
                image = image.float()
            
            return image, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder image if loading fails
            placeholder = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            return placeholder, label

# Legacy function for backward compatibility
def get_transforms(train: bool = True) -> transforms.Compose:
    """Get transforms for training or validation"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
