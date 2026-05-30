#!/usr/bin/env python3
"""Grad-CAM implementation for Plant Disease Detection model (EfficientNet-B2).

Generates heatmaps showing which regions of the leaf image contributed most to the prediction.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import argparse
import json
import os
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping for EfficientNet-B2."""
    
    def __init__(self, model: nn.Module, target_layer: Optional[str] = None):
        """
        Args:
            model: PyTorch model (EfficientNet-B2)
            target_layer: Name of target layer for gradient extraction.
                         If None, uses the final convolutional layer of EfficientNet-B2.
        """
        self.model = model
        self.model.eval()
        
        # Default target layer for EfficientNet-B2 (final conv layer)
        if target_layer is None:
            target_layer = 'features.8.0'  # Final block of EfficientNet-B2
        
        self.target_layer_name = target_layer
        self.target_layer = None
        
        # Find the target layer by name
        for name, layer in model.named_modules():
            if name == target_layer:
                self.target_layer = layer
                break
        
        if self.target_layer is None:
            raise ValueError(f"Target layer '{target_layer}' not found in model")
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        """Hook to save forward activations."""
        self.activations = output
    
    def _save_gradient(self, module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor):
        """Hook to save backward gradients."""
        self.gradients = grad_output[0]
    
    def generate_heatmap(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM heatmap for a given input image.
        
        Args:
            input_tensor: Input image tensor [1, 3, H, W]
            target_class: Target class index. If None, uses predicted class.
        
        Returns:
            Heatmap as numpy array (H, W) normalized to [0, 1]
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        target_score = output[0, target_class]
        target_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(2, 3))
        
        # Weighted combination of activations
        heatmap = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights[0]):
            heatmap += w * activations[0, i, :, :]
        
        # Apply ReLU
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize to [0, 1]
        heatmap_max = heatmap.max()
        if heatmap_max > 0:
            heatmap = heatmap / heatmap_max
        
        return heatmap
    
    def generate_heatmap_batch(self, input_tensor: torch.Tensor, target_classes: Optional[list] = None) -> np.ndarray:
        """Generate Grad-CAM heatmaps for a batch of images.
        
        Args:
            input_tensor: Input image tensor [B, 3, H, W]
            target_classes: List of target class indices. If None, uses predicted classes.
        
        Returns:
            Heatmaps as numpy array [B, H, W] normalized to [0, 1]
        """
        batch_size = input_tensor.shape[0]
        heatmaps = []
        
        for i in range(batch_size):
            img_tensor = input_tensor[i:i+1]
            target_class = target_classes[i] if target_classes else None
            heatmap = self.generate_heatmap(img_tensor, target_class)
            heatmaps.append(heatmap)
        
        return np.array(heatmaps)


def apply_heatmap_to_image(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay heatmap on original image.
    
    Args:
        image: Original image (H, W, 3) in RGB format, values [0, 255]
        heatmap: Heatmap (H, W) with values [0, 1]
        alpha: Transparency of heatmap overlay
    
    Returns:
        Overlay image (H, W, 3)
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Create color heatmap (jet colormap)
    heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend original image and heatmap
    overlay = (image * (1 - alpha) + heatmap_color * alpha).astype(np.uint8)
    
    return overlay


def preprocess_image(img_path: str, image_size: int = 260) -> Tuple[torch.Tensor, np.ndarray]:
    """Preprocess image for Grad-CAM.
    
    Args:
        img_path: Path to image
        image_size: Target size
    
    Returns:
        input_tensor: Tensor for model [1, 3, H, W]
        original_image: Original image as numpy array (H, W, 3) in RGB
    """
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Cannot load image: {img_path}")
    
    # Convert BGR to RGB
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(original_image, (image_size, image_size))
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = (image_normalized - mean) / std
    
    # Convert to tensor
    input_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return input_tensor, original_image


def load_model(model_path: str, num_classes: int = 39) -> nn.Module:
    """Load trained EfficientNet-B2 model.
    
    Args:
        model_path: Path to .pth checkpoint
        num_classes: Number of classes
    
    Returns:
        Loaded model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = models.efficientnet_b2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Linear(in_features, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded from {model_path} to {device}")
    return model, device


def load_labels(labels_path: str = 'labels.json') -> dict:
    """Load class labels."""
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return labels


def save_visualization(original_image: np.ndarray, heatmap: np.ndarray, 
                      overlay: np.ndarray, output_path: str,
                      prediction: str, confidence: float):
    """Save Grad-CAM visualization.
    
    Creates a figure with 3 subplots: original, heatmap, overlay.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Heatmap
    im1 = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f'Prediction: {prediction}\nConfidence: {confidence:.1f}%')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Grad-CAM for Plant Disease Detection')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', default='models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--labels', default='labels.json', help='Path to labels file')
    parser.add_argument('--output', default='gradcam_output.png', help='Output path for visualization')
    parser.add_argument('--target-class', type=int, default=None, help='Target class index (default: predicted class)')
    parser.add_argument('--image-size', type=int, default=260, help='Image size for model input')
    args = parser.parse_args()
    
    # Load model
    model, device = load_model(args.model)
    
    # Load labels
    labels = load_labels(args.labels)
    
    # Create Grad-CAM instance
    grad_cam = GradCAM(model)
    
    # Preprocess image
    input_tensor, original_image = preprocess_image(args.image, args.image_size)
    input_tensor = input_tensor.to(device)
    
    # Generate heatmap
    logger.info("Generating Grad-CAM heatmap...")
    heatmap = grad_cam.generate_heatmap(input_tensor, args.target_class)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_class_idx = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, pred_class_idx].item() * 100
    
    pred_class_name = labels[str(pred_class_idx)] if isinstance(labels, dict) else labels[pred_class_idx]
    
    logger.info(f"Prediction: {pred_class_name} (confidence: {confidence:.1f}%)")
    
    # Apply heatmap to original image
    overlay = apply_heatmap_to_image(original_image, heatmap)
    
    # Save visualization
    save_visualization(original_image, heatmap, overlay, args.output, pred_class_name, confidence)
    
    # Also save individual components
    cv2.imwrite('original.png', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite('overlay.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    logger.info("Individual images saved: original.png, overlay.png")


if __name__ == '__main__':
    main()