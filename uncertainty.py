"""MC Dropout uncertainty estimation.""" 

import torch 
import numpy as np 
from tqdm import tqdm 

def mc_dropout_inference(model, dataloader, num_samples=30, device='cpu'): 
    """Perform MC Dropout inference to estimate uncertainty.""" 
    model.eval() 
    model.enable_mc_dropout() 

    all_predictions = [] 
    all_labels = [] 
    all_entropies = [] 
    all_variances = [] 

    with torch.no_grad(): 
        for images, labels in tqdm(dataloader, desc="MC Inference"): 
            images = images.to(device) 
             
            # Collect predictions for multiple dropout samples 
            sample_predictions = [] 
            for _ in range(num_samples): 
                outputs = model(images).squeeze() 
                probs = torch.sigmoid(outputs) 
                sample_predictions.append(probs.cpu().numpy()) 

            # Stack predictions: (num_samples, batch_size) 
            sample_predictions = np.stack(sample_predictions, axis=0) 

            # Compute statistics 
            mean_probs = np.mean(sample_predictions, axis=0) 
            variance = np.var(sample_predictions, axis=0) 

            # Compute entropy for each sample 
            eps = 1e-8  # Avoid log(0) 
            entropy = -mean_probs * np.log(mean_probs + eps) - (1 - mean_probs) * np.log(1 - mean_probs + eps) 
            
            all_predictions.extend(mean_probs) 
            all_labels.extend(labels.numpy()) 
            all_entropies.extend(entropy) 
            all_variances.extend(variance) 
     
    model.disable_mc_dropout() 
     
    return { 
        'predictions': np.array(all_predictions), 
        'labels': np.array(all_labels), 
        'entropy': np.array(all_entropies), 
        'variance': np.array(all_variances) 
    } 

def compute_uncertainty_features(uncertainty_results): 
    """Extract uncertainty-based features for detection.""" 
    return { 
        'entropy': uncertainty_results['entropy'], 
        'variance': uncertainty_results['variance'], 
        'mean_confidence': uncertainty_results['predictions'] 
    } 

def flag_by_uncertainty(entropy, threshold=0.7): 
    """Flag samples based on entropy threshold.""" 
    return entropy > threshold 