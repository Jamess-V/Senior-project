"""
Visualize dataset images with saliency heatmaps applied.
Displays images from the dataset with Grad-CAM saliency maps overlaid.
"""

import os
import glob
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image
from scipy.ndimage import zoom

from model import MedicalAIModel
from explainability import GradCAMExplainer
from config import Config

# --- Constants -----------------------------------------------------------

IMAGE_SIZE = (244, 244)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
HEATMAP_CMAP = "hot"
OVERLAY_IMAGE_WEIGHT = 0.6
OVERLAY_HEATMAP_WEIGHT = 0.4
POSITIVE_CLASS_THRESHOLD = 0.5
POSITIVE_LABEL = "PNEUMONIA"
NEGATIVE_LABEL = "NORMAL"
FIGURE_DPI = 150
SALIENCY_UPSCALE_ORDER = 3  # cubic interpolation, avoids blocky Grad-CAM cells
UNKNOWN_LABEL = "UNKNOWN"


class SaliencyVisualizer:
    """Visualize images with saliency heatmaps."""

    def __init__(self, model_path: str, model_name: str = "densenet121", device: str = "cpu"):
        """
        Initialize the visualizer.

        Args:
            model_path: Path to the trained model checkpoint.
            model_name: Name of the backbone model.
            device: Device to run inference on ('cpu' or 'cuda').
        """
        self.device = device
        self.model = MedicalAIModel(model_name=model_name, num_classes=1)
        self._load_checkpoint(model_path)

        self.model.to(device)
        self.model.eval()

        self.explainer = GradCAMExplainer(self.model)

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_checkpoint(self, model_path: str) -> None:
        """Load model weights from a checkpoint file, if present."""
        if not os.path.exists(model_path):
            print(f"⚠ Model not found at {model_path}, using untrained model")
            return

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        print(f"✓ Loaded model from {model_path}")

    def get_prediction(self, image_tensor: torch.Tensor) -> float:
        """Get model prediction probability for a single preprocessed image tensor."""
        with torch.no_grad():
            output = self.model(image_tensor.unsqueeze(0).to(self.device))
            prob = torch.sigmoid(output).item()
        return prob

    @staticmethod
    def _label_for(prob: float) -> str:
        return POSITIVE_LABEL if prob > POSITIVE_CLASS_THRESHOLD else NEGATIVE_LABEL

    @staticmethod
    def _ground_truth_for(image_path: str) -> str:
        """Infer the true label from the parent folder name (e.g. .../test/NORMAL/xyz.jpeg)."""
        parent_folder = os.path.basename(os.path.dirname(image_path)).upper()
        if parent_folder in (POSITIVE_LABEL, NEGATIVE_LABEL):
            return parent_folder
        return UNKNOWN_LABEL

    @staticmethod
    def _smooth_saliency(saliency_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Upsample a low-resolution Grad-CAM map to target_size using cubic interpolation.

        Grad-CAM is computed on the last conv layer's feature map (e.g. 7x7 for
        DenseNet121 at 224x224 input), so naively displaying/blending it at full
        resolution produces a harsh, blocky grid. Interpolating first gives a
        smooth heatmap that's much easier to read.
        """
        zoom_factors = (target_size[0] / saliency_map.shape[0], target_size[1] / saliency_map.shape[1])
        smoothed = zoom(saliency_map, zoom_factors, order=SALIENCY_UPSCALE_ORDER)
        return np.clip(smoothed, 0, 1)

    def _analyze_image(self, image_path: str) -> Tuple[Image.Image, np.ndarray, float, str, str]:
        """
        Load an image and compute its (smoothed) saliency map, prediction, and ground truth.

        Returns:
            (original_image, saliency_map, probability, predicted_label, true_label)
        """
        original_image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(original_image)

        prob = self.get_prediction(image_tensor)
        predicted_label = self._label_for(prob)
        true_label = self._ground_truth_for(image_path)

        raw_saliency_map = self.explainer.compute_saliency(image_tensor, target_class=0)
        saliency_map = self._smooth_saliency(raw_saliency_map, IMAGE_SIZE)

        return original_image, saliency_map, prob, predicted_label, true_label

    @staticmethod
    def _make_overlay(original_image: Image.Image, saliency_map: np.ndarray) -> np.ndarray:
        """Blend a resized original image with a colorized saliency heatmap."""
        resized = original_image.resize(IMAGE_SIZE)
        img_array = np.array(resized) / 255.0

        heatmap = plt.colormaps.get_cmap(HEATMAP_CMAP)(saliency_map)
        overlay = OVERLAY_IMAGE_WEIGHT * img_array + OVERLAY_HEATMAP_WEIGHT * heatmap[:, :, :3]
        return np.clip(overlay, 0, 1)

    def _plot_triplet(self, axes_row, original_image, saliency_map, prob, predicted_label,
                       true_label, image_title: str) -> None:
        """Render original / saliency / overlay into a row of three axes."""
        axes_row[0].imshow(original_image)
        axes_row[0].set_title(image_title, fontsize=11, fontweight="bold")
        axes_row[0].axis("off")

        im = axes_row[1].imshow(saliency_map, cmap=HEATMAP_CMAP)
        axes_row[1].set_title("Saliency Map (Grad-CAM)", fontsize=11, fontweight="bold")
        axes_row[1].axis("off")
        plt.colorbar(im, ax=axes_row[1], fraction=0.046, pad=0.04)

        overlay = self._make_overlay(original_image, saliency_map)
        axes_row[2].imshow(overlay)

        is_correct = true_label != UNKNOWN_LABEL and predicted_label == true_label
        title_color = "black" if true_label == UNKNOWN_LABEL else ("darkgreen" if is_correct else "darkred")
        overlay_title = f"True: {true_label} | Pred: {predicted_label} ({prob:.2%})"
        axes_row[2].set_title(overlay_title, fontsize=11, fontweight="bold", color=title_color)
        axes_row[2].axis("off")

    def visualize_image_with_saliency(self, image_path: str, save_path: Optional[str] = None) -> Figure:
        """
        Create a visualization of one image with its saliency heatmap.

        Args:
            image_path: Path to the image.
            save_path: Optional path to save the visualization.

        Returns:
            The matplotlib Figure.
        """
        original_image, saliency_map, prob, predicted_label, true_label = self._analyze_image(image_path)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        self._plot_triplet(axes, original_image, saliency_map, prob, predicted_label, true_label,
                            image_title="Original Image")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
            print(f"✓ Saved visualization to {save_path}")

        return fig

    def visualize_batch(
        self,
        image_paths: Sequence[str],
        output_dir: str = "outputs/figures/saliency",
        show: bool = True,
    ) -> Figure:
        """
        Create a grid visualization of multiple images with saliency maps.

        Each row shows one image as: original, saliency map, overlay.

        Args:
            image_paths: List of image file paths to visualize (one row each).
            output_dir: Directory to save the visualization.
            show: Whether to call plt.show() after rendering.

        Returns:
            The matplotlib Figure.
        """
        os.makedirs(output_dir, exist_ok=True)

        valid_paths = [p for p in image_paths if os.path.exists(p)]
        missing = set(image_paths) - set(valid_paths)
        for path in missing:
            print(f"⚠ Image not found: {path}")

        if not valid_paths:
            raise ValueError("No valid image paths were provided for batch visualization.")

        n_rows = len(valid_paths)
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, image_path in enumerate(valid_paths):
            original_image, saliency_map, prob, predicted_label, true_label = self._analyze_image(image_path)
            self._plot_triplet(
                axes[idx], original_image, saliency_map, prob, predicted_label, true_label,
                image_title=f"Image {idx + 1}",
            )

        plt.tight_layout()

        output_path = os.path.join(output_dir, "saliency_visualization.png")
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        print(f"✓ Saved batch visualization to {output_path}")

        if show:
            plt.show()

        return fig


def get_sample_images(data_root: str, n_normal: int = 2, n_pneumonia: int = 2) -> List[str]:
    """Collect a random sample of test images from each class."""
    normal_images = glob.glob(os.path.join(data_root, "test", "NORMAL", "*.jpeg"))
    pneumonia_images = glob.glob(os.path.join(data_root, "test", "PNEUMONIA", "*.jpeg"))
    normal_sample = random.sample(normal_images, min(n_normal, len(normal_images)))
    pneumonia_sample = random.sample(pneumonia_images, min(n_pneumonia, len(pneumonia_images)))
    return normal_sample + pneumonia_sample


def main():
    """Main function to run saliency visualization."""
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = "outputs/checkpoints/best_model.pth"
    visualizer = SaliencyVisualizer(model_path=model_path, model_name=config.model_name, device=device)

    sample_images = get_sample_images(config.data_root)

    print(f"\n{'=' * 60}")
    print("Generating Saliency Visualizations")
    print(f"{'=' * 60}\n")

    if not sample_images:
        print("❌ No images found in dataset!")
        print(f"Expected dataset at: {config.data_root}")
        return

    visualizer.visualize_batch(sample_images)

    individual_dir = "outputs/figures/saliency/individual"
    os.makedirs(individual_dir, exist_ok=True)
    for idx, image_path in enumerate(sample_images):
        save_path = os.path.join(individual_dir, f"saliency_{idx + 1}.png")
        visualizer.visualize_image_with_saliency(image_path, save_path=save_path)
        plt.close()


if __name__ == "__main__":
    main()