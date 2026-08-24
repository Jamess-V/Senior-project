"""Configuration parameters for the project.""" 

import os 
from dataclasses import dataclass 
from typing import Tuple 

@dataclass 
class Config: 

    # Data paths 
    data_root = "./data/chestxray2017/chest_xray"
    output_dir: str = "./outputs" 

    # Image parameters 
    image_size: Tuple[int, int] = (224, 224) 
    num_classes: int = 1  # Binary classification (disease present/absent) 

    # Model parameters 
    model_name: str = "densenet121"  # Options: 'densenet121', 'resnet50', 'efficientnet_b0' 
    pretrained: bool = True 

    # MC Dropout parameters 
    mc_dropout_samples: int = 30 
    mc_dropout_keep_prob: float = 0.5  

    # Training parameters 
    batch_size: int = 32 
    num_epochs: int = 10 
    learning_rate: float = 1e-4 
    weight_decay: float = 1e-5 

    # Detection parameters 
    uncertainty_threshold: float = 0.7   # Entropy threshold for flagging 
    plausibility_threshold: float = 0.35  # Minimum saliency-lung overlap 

    # Evaluation parameters 
    k_values_for_precision: Tuple[int, ...] = (10, 20, 50) 
 
    # Device 
    device: str = "cpu"  # 'cuda' or 'cpu' 

    # Random seed 
    seed: int = 42 

    def __post_init__(self): 
        os.makedirs(self.output_dir, exist_ok=True) 
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True) 
        os.makedirs(os.path.join(self.output_dir, "figures"), exist_ok=True) 
        os.makedirs(os.path.join(self.output_dir, "results"), exist_ok=True) 

config = Config() 

 