"""Evaluation metrics and reporting."""  

import numpy as np 
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix 
import matplotlib.pyplot as plt 
import seaborn as sns 
import json 
import os 
from detector import MisleadingOutputDetector

def precision_at_k(scores, true_labels, k_values): 
    """Calculate Precision@K for multiple K values.""" 
    results = {} 

    # Sort indices by score descending 
    sorted_indices = np.argsort(scores)[::-1] 
    sorted_labels = true_labels[sorted_indices] 

    for k in k_values: 
        k = min(k, len(sorted_labels)) 
        top_k_labels = sorted_labels[:k] 
        precision = np.sum(top_k_labels) / k if k > 0 else 0 
        results[f'precision@{k}'] = precision 

    return results 

def recall_at_k(scores, true_labels, k_values): 
    """Calculate Recall@K (coverage of true positives in top K).""" 
    results = {} 
    total_positives = np.sum(true_labels) 

    sorted_indices = np.argsort(scores)[::-1] 
    sorted_labels = true_labels[sorted_indices] 

    for k in k_values: 
        k = min(k, len(sorted_labels)) 
        top_k_labels = sorted_labels[:k] 
        recall = np.sum(top_k_labels) / total_positives if total_positives > 0 else 0 
        results[f'recall@{k}'] = recall 

    return results 

def compute_all_metrics(detection_results, config): 
    """Compute all evaluation metrics.""" 
    scores = detection_results['detection_scores'] 
    ground_truth = detection_results.get('ground_truth') 

    # If ground truth not provided, create proxy 
    if ground_truth is None: 
        predictions = detection_results['predictions'] 
        labels = detection_results['labels'] 
        plausibility = detection_results['plausibility'] 

        detector = MisleadingOutputDetector() 
        ground_truth = detector.get_ground_truth( 
            predictions, labels,  
            correctness=(predictions > 0.5) == labels, 
            plausibility_scores=plausibility
        ) 
        detection_results['ground_truth'] = ground_truth 

    # Compute metrics 
    metrics = {}
    
    # AUROC 
    if len(np.unique(ground_truth)) > 1: 
        metrics['auroc'] = roc_auc_score(ground_truth, scores) 
    else: 
        metrics['auroc'] = 0.5 

    # Precision@K 
    k_values = config.k_values_for_precision 
    precision_results = precision_at_k(scores, ground_truth, k_values) 
    metrics.update(precision_results) 

    # Recall@K 
    recall_results = recall_at_k(scores, ground_truth, k_values) 
    metrics.update(recall_results) 

    # Best threshold (Youden's J index) 
    fpr, tpr, thresholds = roc_curve(ground_truth, scores) 
    youden_idx = np.argmax(tpr - fpr) 
    metrics['best_threshold'] = thresholds[youden_idx] if len(thresholds) > 0 else 0.5 
    metrics['best_tpr'] = tpr[youden_idx] if len(tpr) > 0 else 0 
    metrics['best_fpr'] = fpr[youden_idx] if len(fpr) > 0 else 0 

    return metrics, ground_truth  

def plot_evaluation_results(detection_results, metrics, config): 
    """Generate evaluation plots.""" 
    scores = detection_results['detection_scores'] 
    ground_truth = detection_results['ground_truth'] 

    output_dir = os.path.join(config.output_dir, "figures")
    
    # 1. ROC Curve 
    if len(np.unique(ground_truth)) > 1: 
        fpr, tpr, _ = roc_curve(ground_truth, scores) 
        plt.figure(figsize=(8, 6)) 
        plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['auroc']:.3f})") 
        plt.plot([0, 1], [0, 1], 'k--', label="Random") 
        plt.xlabel("False Positive Rate") 
        plt.ylabel("True Positive Rate") 
        plt.title("ROC Curve - Misleading Output Detection") 
        plt.legend() 
        plt.grid(True) 
        plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300, bbox_inches='tight') 
        plt.close() 

    # 2. Precision-Recall Curve 
    precision, recall, _ = precision_recall_curve(ground_truth, scores) 
    plt.figure(figsize=(8, 6)) 
    plt.plot(recall, precision, label="PR Curve") 
    plt.xlabel("Recall") 
    plt.ylabel("Precision") 
    plt.title("Precision-Recall Curve") 
    plt.grid(True) 
    plt.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=300, bbox_inches='tight') 
    plt.close() 

    # 3. Distribution of detection scores 
    plt.figure(figsize=(10, 6)) 
    plt.hist(scores[ground_truth == 1], bins=30, alpha=0.7, label="Truly Misleading", color='red') 
    plt.hist(scores[ground_truth == 0], bins=30, alpha=0.7, label="Reliable", color='green') 
    plt.xlabel("Detection Score") 
    plt.ylabel("Count") 
    plt.title("Distribution of Detection Scores") 
    plt.legend() 
    plt.grid(True, alpha=0.3) 
    plt.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=300, bbox_inches='tight') 
    plt.close() 

    # 4. Precision@K bar chart 
    k_values = [int(k.split('@')[1]) for k in metrics.keys() if k.startswith('precision@')] 
    precision_values = [metrics[f'precision@{k}'] for k in k_values] 

    plt.figure(figsize=(8, 6)) 
    plt.bar(k_values, precision_values, color='skyblue') 
    plt.xlabel("K") 
    plt.ylabel("Precision@K") 
    plt.title("Precision@K Evaluation") 
    plt.ylim(0, 1) 
    for i, v in enumerate(precision_values): 
        plt.text(k_values[i], v + 0.02, f"{v:.3f}", ha='center') 
    plt.grid(True, alpha=0.3) 
    plt.savefig(os.path.join(output_dir, "precision_at_k.png"), dpi=300, bbox_inches='tight') 
    plt.close() 