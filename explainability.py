"""Explainability methods: Grad-CAM and plausibility scoring.""" 

 

import torch 

import torch.nn.functional as F 

import numpy as np 

import cv2 

from captum.attr import LayerGradCam, LayerAttribution 

import matplotlib.pyplot as plt 

 

class GradCAMExplainer: 

    """Grad-CAM explainability wrapper.""" 

     

    def __init__(self, model, target_layer=None): 

        self.model = model 

        self.target_layer = target_layer or self._find_last_conv_layer() 

        self.grad_cam = LayerGradCam(model, self.target_layer) 

     

    def _find_last_conv_layer(self): 

        """Find the last convolutional layer in the backbone.""" 

        backbone = self.model.backbone 

         

        # For DenseNet 

        if hasattr(backbone, 'features'): 

            return backbone.features[-1] 

         

        # For ResNet 

        if hasattr(backbone, 'layer4'): 

            return backbone.layer4[-1] 

         

        # For EfficientNet 

        if hasattr(backbone, 'features'): 

            # Get last conv in EfficientNet 

            layers = list(backbone.features.children()) 

            for layer in reversed(layers): 

                if isinstance(layer, torch.nn.Conv2d): 

                    return layer 

         

        raise ValueError("Could not find convolutional layer") 

     

    def compute_saliency(self, image, target_class=0): 

        """Compute Grad-CAM saliency map.""" 

        self.model.eval() 

         

        # Image should be [1, C, H, W] 

        if len(image.shape) == 3: 

            image = image.unsqueeze(0) 

         

        # Compute Grad-CAM attributions 

        attributions = self.grad_cam.attribute(image, target=target_class) 

         

        # Upsample to original image size 

        attributions = LayerAttribution.interpolate(attributions,  

                                                   (image.shape[2], image.shape[3])) 

         

        # Convert to numpy 

        saliency = attributions.squeeze().cpu().detach().numpy() 

         

        # Normalize to [0, 1] 

        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8) 

         

        return saliency 

     

    def compute_lung_overlap(self, saliency, lung_mask): 

        """Compute overlap between saliency map and lung region.""" 

        # Convert saliency to binary mask (top 50% of activations) 

        threshold = np.percentile(saliency, 50) 

        saliency_binary = (saliency > threshold).astype(np.float32) 

         

        # Compute intersection over union 

        intersection = np.sum(saliency_binary * lung_mask) 

        union = np.sum(saliency_binary + lung_mask > 0) 

         

        if union == 0: 

            return 0.0 

         

        iou = intersection / union 

         

        # Also compute percentage of saliency in lung region 

        lung_activation = np.sum(saliency * lung_mask) / (np.sum(saliency) + 1e-8) 

         

        return lung_activation, iou 

 

def generate_saliency_maps(model, dataloader, device, num_samples=50): 

    """Generate saliency maps for a subset of samples.""" 

    explainer = GradCAMExplainer(model) 

     

    saliency_results = []
    processed_samples = 0

     

    model.eval() 

    with torch.enable_grad():

        for i, (images, labels) in enumerate(dataloader): 

            if processed_samples >= num_samples:

                break 

             

            images = images.to(device) 

            labels = labels.cpu().numpy() 

             

            for j in range(images.size(0)):
                if processed_samples >= num_samples:
                    break

                image = images[j] 

                label = labels[j] 

                 

                # Forward pass to get prediction 

                output = model(image.unsqueeze(0)).flatten()[0]

                prediction = torch.sigmoid(output).item() 

                pred_label = 1 if prediction > 0.5 else 0 

                 

                # Compute saliency for predicted class (or target if misclassified) 

                target_class = 0  # Use predicted class 

                 

                saliency = explainer.compute_saliency(image, target_class=target_class) 

                 

                # Create dummy lung mask (in practice, you'd use a segmentation model) 

                # For demo: assume central region is lung (will be replaced with real masks) 

                lung_mask = np.ones_like(saliency) 

                h, w = saliency.shape 

                center_h, center_w = h//2, w//2 

                lung_mask[:center_h//2, :] = 0 

                lung_mask[3*center_h//2:, :] = 0 

                 

                lung_activation, iou = explainer.compute_lung_overlap(saliency, lung_mask) 

                 

                saliency_results.append({ 

                    'image': image.cpu().numpy(), 

                    'label': label, 

                    'prediction': prediction, 

                    'pred_label': pred_label, 

                    'is_correct': (label == pred_label), 

                    'saliency': saliency, 

                    'lung_activation': lung_activation, 

                    'iou': iou 

                })
                processed_samples += 1

     

    return saliency_results 

 

def compute_plausibility_features(saliency_results): 

    """Extract plausibility features from saliency results.""" 

    return { 

        'lung_activation': np.array([r['lung_activation'] for r in saliency_results]), 

        'iou': np.array([r['iou'] for r in saliency_results]), 

        'is_correct': np.array([r['is_correct'] for r in saliency_results]), 

        'prediction': np.array([r['prediction'] for r in saliency_results]), 

        'label': np.array([r['label'] for r in saliency_results]) 

    } 