#!/usr/bin/env python3
"""Grad-CAM implementation for the EfficientNet-B2 plant disease model.

Generates heatmaps showing which image regions contributed most to a prediction.
This is an offline/Python explainability tool; the Android app uses forward-only
TorchScript inference and does not run true Grad-CAM on device.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import argparse
import json
from pathlib import Path
from typing import Optional, Tuple
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
            target_layer = 'features.8'  # Final EfficientNet-B2 convolution path
        
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
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = (image_normalized - mean) / std
    
    # Convert to tensor
    input_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0).float()
    
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
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded from %s to %s", model_path, device)
    return model, device


def load_labels(labels_path: str = 'labels.json') -> dict:
    """Load class labels."""
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    return labels


def resolve_default_checkpoint(project_root: Path) -> Path:
    """Return the preferred fine-tuned checkpoint for Grad-CAM."""
    preferred = project_root / "models" / "best_model_finetuned.pth"
    if preferred.exists():
        return preferred

    candidates = sorted(
        (path for path in (project_root / "models").glob("best_model_finetuned*.pth") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        selected = candidates[0]
        logger.info(
            "Default checkpoint %s was not found; using newest fine-tuned checkpoint: %s",
            preferred,
            selected,
        )
        return selected

    baseline = project_root / "models" / "best_model.pth"
    if baseline.exists():
        logger.warning("No fine-tuned checkpoint found; falling back to baseline checkpoint: %s", baseline)
        return baseline

    raise FileNotFoundError(f"No model checkpoint found under {project_root / 'models'}")


def label_name(labels, index: int) -> str:
    if isinstance(labels, dict):
        return labels.get(str(index), labels.get(index, f"class_{index}"))
    return labels[index]


def top_k_predictions(output: torch.Tensor, labels, k: int = 3) -> list[tuple[int, str, float]]:
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    top_count = min(k, probabilities.numel())
    confidences, indices = torch.topk(probabilities, top_count)
    return [
        (int(index.item()), label_name(labels, int(index.item())), float(conf.item()) * 100.0)
        for conf, index in zip(confidences, indices)
    ]


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
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description='Grad-CAM for Plant Disease Detection')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument(
        '--model',
        default=None,
        help='Path to model checkpoint. Defaults to the newest fine-tuned checkpoint.',
    )
    parser.add_argument('--labels', default=str(project_root / 'labels.json'), help='Path to labels file')
    parser.add_argument('--output', default=str(project_root / 'reports' / 'gradcam' / 'gradcam_output.png'), help='Output path for visualization')
    parser.add_argument('--target-class', type=int, default=None, help='Target class index (default: predicted class)')
    parser.add_argument('--target-layer', default=None, help='Model layer used for Grad-CAM hooks. Default: features.8')
    parser.add_argument('--image-size', type=int, default=260, help='Image size for model input')
    args = parser.parse_args()
    model_path = args.model or str(resolve_default_checkpoint(project_root))
    
    # Load model
    model, device = load_model(model_path)
    
    # Load labels
    labels = load_labels(args.labels)
    
    # Create Grad-CAM instance
    grad_cam = GradCAM(model, target_layer=args.target_layer)
    
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
    
    pred_class_name = label_name(labels, pred_class_idx)
    
    logger.info(f"Prediction: {pred_class_name} (confidence: {confidence:.1f}%)")
    logger.info("Top-3 predictions:")
    for class_index, class_name, score in top_k_predictions(output, labels, k=3):
        logger.info("  %s (%d): %.2f%%", class_name, class_index, score)
    
    # Apply heatmap to original image
    overlay = apply_heatmap_to_image(original_image, heatmap)
    
    # Save visualization
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_visualization(original_image, heatmap, overlay, str(output_path), pred_class_name, confidence)
    
    # Also save individual components
    original_path = output_path.with_name(f"{output_path.stem}_original.png")
    overlay_path = output_path.with_name(f"{output_path.stem}_overlay.png")
    cv2.imwrite(str(original_path), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    logger.info("Individual images saved: %s, %s", original_path, overlay_path)


if __name__ == '__main__':
    main()
