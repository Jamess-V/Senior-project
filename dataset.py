"""Dataset loading and preprocessing for chest X-ray images.""" 

import glob
import os 
import torch 
from torch.utils.data import Dataset, DataLoader 
from torchvision import transforms 
from PIL import Image 
import numpy as np 
from sklearn.model_selection import train_test_split 
 
class ChestXRayDataset(Dataset): 
    """Custom dataset for chest X-ray classification."""     

    def __init__(self, image_paths, labels, transform=None): 
        self.image_paths = image_paths 
        self.labels = labels 
        self.transform = transform   

    def __len__(self): 
        return len(self.image_paths) 

    def __getitem__(self, idx): 

        image = Image.open(self.image_paths[idx]).convert('RGB') 

        label = self.labels[idx] 

         

        if self.transform: 

            image = self.transform(image) 

         

        return image, torch.tensor(label, dtype=torch.float32) 

 

def get_transforms(image_size=(224, 224), is_train=True): 

    """Get data augmentation transforms.""" 

    if is_train: 

        transform = transforms.Compose([ 

            transforms.Resize(image_size), 

            transforms.RandomHorizontalFlip(p=0.5), 

            transforms.RandomRotation(10), 

            transforms.ColorJitter(brightness=0.1, contrast=0.1), 

            transforms.ToTensor(), 

            transforms.Normalize(mean=[0.485, 0.456, 0.406],  

                               std=[0.229, 0.224, 0.225]) 

        ]) 

    else: 

        transform = transforms.Compose([ 

            transforms.Resize(image_size), 

            transforms.ToTensor(), 

            transforms.Normalize(mean=[0.485, 0.456, 0.406],  

                               std=[0.229, 0.224, 0.225]) 

        ]) 

     

    return transform 

 

def load_data(data_root, test_size=0.2, val_size=0.1, random_state=42): 

    """Load dataset and create train/val/test splits. 

     

    Expected directory structure: 

    data_root/ 

    ├── train/ 

    │   ├── NORMAL/ 

    │   └── PNEUMONIA/ 

    └── test/ 

        ├── NORMAL/ 

        └── PNEUMONIA/ 

    """ 

    # This is a placeholder - you'll need to adapt to your actual dataset 

    # For now, we simulate data paths 

     

    # In practice, you would use os.walk to collect all image paths 

    # and assign labels based on directory names 

     

    print("Loading data from:", data_root) 

     

    # Placeholder: create dummy data for demonstration 

    # Replace this with actual data loading logic 

    from glob import glob

    extensions = ["*.jpg", "*.jpeg", "*.png"]

    normal_paths = []
    pneumonia_paths = []

    for ext in extensions:
        normal_paths.extend(
            glob(os.path.join(data_root, "**", "NORMAL", ext), recursive=True)
        )

        pneumonia_paths.extend(
            glob(os.path.join(data_root, "**", "PNEUMONIA", ext), recursive=True)
        )
     

    image_paths = normal_paths + pneumonia_paths

    labels = [0] * len(normal_paths) + [1] * len(pneumonia_paths)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under {data_root}")

     

    # Split data 

    official_train_root = os.path.abspath(os.path.join(data_root, "train"))
    official_test_root = os.path.abspath(os.path.join(data_root, "test"))
    official_val_root = os.path.abspath(os.path.join(data_root, "val"))
    has_official_split = os.path.isdir(official_train_root) and os.path.isdir(official_test_root)

    if has_official_split:
        X_train = [path for path in image_paths if os.path.commonpath([os.path.abspath(path), official_train_root]) == official_train_root]
        X_test = [path for path in image_paths if os.path.commonpath([os.path.abspath(path), official_test_root]) == official_test_root]
        y_train = [1 if os.path.basename(os.path.dirname(path)) == "PNEUMONIA" else 0 for path in X_train]
        y_test = [1 if os.path.basename(os.path.dirname(path)) == "PNEUMONIA" else 0 for path in X_test]
        if os.path.isdir(official_val_root):
            X_val = [path for path in image_paths if os.path.commonpath([os.path.abspath(path), official_val_root]) == official_val_root]
            y_val = [1 if os.path.basename(os.path.dirname(path)) == "PNEUMONIA" else 0 for path in X_val]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            image_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size / (1 - test_size),
            random_state=random_state, stratify=y_train
        )

    print(f"Found {len(image_paths)} images")
    print(f"Found {len(labels)} labels")

     

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}") 

     

    # Create datasets 

    train_dataset = ChestXRayDataset( 

        X_train, y_train, get_transforms(is_train=True) 

    ) 

    val_dataset = ChestXRayDataset( 

        X_val, y_val, get_transforms(is_train=False) 

    ) 

    test_dataset = ChestXRayDataset( 

        X_test, y_test, get_transforms(is_train=False) 

    ) 

     

    return train_dataset, val_dataset, test_dataset 

 

def get_dataloaders(config, train_dataset, val_dataset, test_dataset): 

    """Create DataLoaders.""" 

    train_loader = DataLoader( 

        train_dataset, batch_size=config.batch_size,  

        shuffle=True, num_workers=4, pin_memory=True 

    ) 

    val_loader = DataLoader( 

        val_dataset, batch_size=config.batch_size, 

        shuffle=False, num_workers=4, pin_memory=True 

    ) 

    test_loader = DataLoader( 

        test_dataset, batch_size=config.batch_size, 

        shuffle=False, num_workers=4, pin_memory=True 

    ) 

     

    return train_loader, val_loader, test_loader 

 