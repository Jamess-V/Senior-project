"""Main pipeline runner for training, uncertainty estimation, explainability, and evaluation."""

import json
import os
import random

import numpy as np 
import torch
from torch.utils.data import DataLoader, Subset

from config import config
from dataset import load_data, get_dataloaders
from evaluate import compute_all_metrics, plot_evaluation_results
from explainability import compute_plausibility_features, generate_saliency_maps
from model import create_model
from train import train_model, validate
from uncertainty import mc_dropout_inference


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_scores(scores):
    """Min-max normalize a NumPy array to the [0, 1] range."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    min_val = scores.min()
    max_val = scores.max()
    if np.isclose(max_val, min_val):
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val + 1e-8)


def main():
    """Run the full pipeline."""
    print("=" * 60)
    print("Medical AI Safety - Misleading Output Detection")
    print("=" * 60)

    set_seed(config.seed)

    print("\n[1/6] Loading data...")
    train_dataset, val_dataset, test_dataset = load_data(
        config.data_root,
        test_size=0.2,
        val_size=0.1,
        random_state=config.seed,
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        config,
        train_dataset,
        val_dataset,
        test_dataset,
    )

    print("\n[2/6] Creating model...")
    model = create_model(config)

    print("\n[3/6] Training model...")
    model, history = train_model(model, train_loader, val_loader, config)

    print("\n[4/6] Running final evaluation on test data...")
    test_criterion = torch.nn.BCEWithLogitsLoss()
    test_loss, test_acc = validate(model, test_loader, test_criterion, config.device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    print("\n[5/6] Running uncertainty and explainability analysis...")
    sample_size = min(len(test_dataset), 256)
    subset_indices = list(range(sample_size))
    subset_dataset = Subset(test_dataset, subset_indices)
    subset_loader = DataLoader(
        subset_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    uncertainty_results = mc_dropout_inference(
        model,
        subset_loader,
        num_samples=config.mc_dropout_samples,
        device=config.device,
    )
    saliency_results = generate_saliency_maps(
        model,
        subset_loader,
        device=config.device,
        num_samples=sample_size,
    )
    plausibility_features = compute_plausibility_features(saliency_results)

    entropy = _normalize_scores(uncertainty_results["entropy"])
    lung_activation = _normalize_scores(plausibility_features["lung_activation"])
    iou = _normalize_scores(plausibility_features["iou"])
    plausibility_score = np.clip((lung_activation + iou) / 2.0, 0.0, 1.0)
    detection_scores = np.clip(0.65 * entropy + 0.35 * (1.0 - plausibility_score), 0.0, 1.0)

    detection_results = {
        "detection_scores": detection_scores,
        "predictions": uncertainty_results["predictions"],
        "labels": uncertainty_results["labels"],
        "plausibility": plausibility_score,
        "ground_truth": None,
    }

    print("\n[6/6] Computing detection metrics and plots...")
    ground_truth = detection_results.get("ground_truth")
    metrics = {}
    if ground_truth is None:
        print(
            "No independent misleading-output labels supplied; "
            "skipping detection metrics."
        )
    else:
        metrics, ground_truth = compute_all_metrics(detection_results, config)
        plot_evaluation_results(detection_results, metrics, config)

    summary = {
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "metrics": metrics,
        "ground_truth": None if ground_truth is None else ground_truth.tolist(),
        "history": history,
    }
    summary_path = os.path.join(config.output_dir, "results", "pipeline_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print("\nPipeline complete.")
    print(f"Saved results to: {summary_path}")
    print("Best threshold:", metrics.get("best_threshold"))
    print("AUROC:", metrics.get("auroc"))


if __name__ == "__main__":
    main()