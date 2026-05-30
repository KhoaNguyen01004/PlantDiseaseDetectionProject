#!/usr/bin/env python3
"""Metadata and model versioning utilities for Plant Disease Detection.

Provides unified metadata export for model deployment.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelMetadata:
    """Manages model metadata for deployment."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize metadata manager.
        
        Args:
            config: Configuration dictionary. If None, uses defaults.
        """
        self.config = config or {}
        self.metadata = {
            "model": {
                "architecture": self.config.get("architecture", "efficientnet_b2"),
                "num_classes": self.config.get("num_classes", 39),
                "image_size": self.config.get("image_size", 260),
                "input_mean": [0.485, 0.456, 0.406],
                "input_std": [0.229, 0.224, 0.225],
            },
            "training": {
                "dataset": self.config.get("dataset", "PlantVillage"),
                "epochs": self.config.get("epochs", 20),
                "batch_size": self.config.get("batch_size", 32),
                "learning_rate": self.config.get("learning_rate", 1e-4),
            },
            "version": {
                "model": self.config.get("version", "1.0.0"),
                "export_date": datetime.now().isoformat(),
                "export_tool": "plant_disease_detection_v1",
            },
            "deployment": {
                "target_platform": "android",
                "tflite_compatible": True,
                "min_android_api": 24,
            },
            "labels": {}  # Will be filled by export_labels()
        }
    
    def set_labels(self, label_map: Dict[int, str]):
        """Set class labels in metadata.
        
        Args:
            label_map: Dictionary mapping class indices to class names.
        """
        self.metadata["labels"] = {
            str(k): v for k, v in label_map.items()
        }
        self.metadata["model"]["num_classes"] = len(label_map)
    
    def set_training_metrics(self, metrics: Dict[str, float]):
        """Set training metrics in metadata.
        
        Args:
            metrics: Dictionary of training metrics (accuracy, loss, etc.)
        """
        self.metadata["training"]["metrics"] = metrics
    
    def export_metadata(self, output_dir: str = ".") -> str:
        """Export metadata to JSON file.
        
        Args:
            output_dir: Directory to save metadata file.
        
        Returns:
            Path to exported metadata file.
        """
        os.makedirs(output_dir, exist_ok=True)
        metadata_path = os.path.join(output_dir, "metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Metadata exported to {metadata_path}")
        return metadata_path
    
    def export_labels_json(self, output_dir: str = ".", label_map: Optional[Dict[int, str]] = None) -> str:
        """Export labels to JSON file.
        
        Args:
            output_dir: Directory to save labels file.
            label_map: Optional label map to export. If None, uses existing labels.
        
        Returns:
            Path to exported labels file.
        """
        if label_map:
            self.set_labels(label_map)
        
        os.makedirs(output_dir, exist_ok=True)
        labels_path = os.path.join(output_dir, "labels.json")
        
        # Export as {index: label} format
        labels_dict = {str(k): v for k, v in self.metadata["labels"].items()}
        
        with open(labels_path, 'w') as f:
            json.dump(labels_dict, f, indent=2)
        
        logger.info(f"Labels exported to {labels_path}")
        return labels_path
    
    def export_labels_txt(self, output_dir: str = ".", label_map: Optional[Dict[int, str]] = None) -> str:
        """Export labels to text file (one label per line).
        
        Args:
            output_dir: Directory to save labels file.
            label_map: Optional label map to export. If None, uses existing labels.
        
        Returns:
            Path to exported labels file.
        """
        if label_map:
            self.set_labels(label_map)
        
        os.makedirs(output_dir, exist_ok=True)
        labels_path = os.path.join(output_dir, "labels.txt")
        
        # Sort by index and write one per line
        sorted_labels = sorted(self.metadata["labels"].items(), key=lambda x: int(x[0]))
        
        with open(labels_path, 'w') as f:
            for _, label in sorted_labels:
                f.write(label + '\n')
        
        logger.info(f"Labels exported to {labels_path}")
        return labels_path
    
    def export_all(self, output_dir: str = ".", label_map: Optional[Dict[int, str]] = None) -> Dict[str, str]:
        """Export all metadata files (metadata.json, labels.json, labels.txt).
        
        Args:
            output_dir: Directory to save files.
            label_map: Optional label map to export.
        
        Returns:
            Dictionary of exported file paths.
        """
        paths = {}
        
        if label_map:
            self.set_labels(label_map)
        
        paths["metadata"] = self.export_metadata(output_dir)
        paths["labels_json"] = self.export_labels_json(output_dir)
        paths["labels_txt"] = self.export_labels_txt(output_dir)
        
        logger.info(f"All metadata files exported to {output_dir}")
        logger.info(f"  - {paths['metadata']}")
        logger.info(f"  - {paths['labels_json']}")
        logger.info(f"  - {paths['labels_txt']}")
        
        return paths
    
    def load_metadata(self, metadata_path: str) -> Dict:
        """Load metadata from JSON file.
        
        Args:
            metadata_path: Path to metadata JSON file.
        
        Returns:
            Metadata dictionary.
        """
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        logger.info(f"Metadata loaded from {metadata_path}")
        return self.metadata
    
    def get_label_map(self) -> Dict[int, str]:
        """Get label map from metadata.
        
        Returns:
            Dictionary mapping class indices (int) to class names (str).
        """
        return {int(k): v for k, v in self.metadata["labels"].items()}
    
    def get_label_list(self) -> List[str]:
        """Get list of labels in order.
        
        Returns:
            List of class names in order of their indices.
        """
        sorted_labels = sorted(self.metadata["labels"].items(), key=lambda x: int(x[0]))
        return [label for _, label in sorted_labels]
    
    def __str__(self) -> str:
        """String representation of metadata."""
        return json.dumps(self.metadata, indent=2)


def export_metadata_for_model(
    label_map: Dict[int, str],
    output_dir: str = ".",
    config: Optional[Dict] = None,
    metrics: Optional[Dict[str, float]] = None
) -> Dict[str, str]:
    """Convenience function to export all metadata for a trained model.
    
    Args:
        label_map: Dictionary mapping class indices to class names.
        output_dir: Directory to save metadata files.
        config: Optional configuration dictionary.
        metrics: Optional training metrics.
    
    Returns:
        Dictionary of exported file paths.
    """
    metadata_manager = ModelMetadata(config)
    
    if metrics:
        metadata_manager.set_training_metrics(metrics)
    
    return metadata_manager.export_all(output_dir, label_map)


def main():
    """Example usage of metadata export."""
    parser = argparse.ArgumentParser(description='Export model metadata')
    parser.add_argument('--output-dir', default='.', help='Output directory for metadata files')
    parser.add_argument('--labels', default='labels.json', help='Path to labels JSON file')
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    import yaml
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Load labels
    label_map = {}
    if os.path.exists(args.labels):
        with open(args.labels, 'r') as f:
            labels_dict = json.load(f)
            label_map = {int(k): v for k, v in labels_dict.items()}
    
    # Export metadata
    if label_map:
        paths = export_metadata_for_model(label_map, args.output_dir, config)
        print(f"\nMetadata exported:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    else:
        print("No labels found. Please provide a valid labels.json file.")


if __name__ == '__main__':
    import argparse
    main()