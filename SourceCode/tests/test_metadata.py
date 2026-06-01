import os
import json
import pytest
from src.metadata import ModelMetadata, export_metadata_for_model

def test_model_metadata_init(sample_config):
    """Test ModelMetadata class instantiation and default structure."""
    meta = ModelMetadata(sample_config)
    assert isinstance(meta.metadata, dict)
    assert meta.metadata["model"]["architecture"] == "efficientnet_b2"
    assert meta.metadata["model"]["image_size"] == 260
    assert meta.metadata["training"]["dataset_version"] == "v1.0.0"

def test_set_labels_and_metrics(sample_config):
    """Test setting labels and training metrics on metadata."""
    meta = ModelMetadata(sample_config)
    
    label_map = {0: "Apple___healthy", 1: "Tomato___healthy"}
    meta.set_labels(label_map)
    assert meta.metadata["model"]["num_classes"] == 2
    assert meta.metadata["labels"]["0"] == "Apple___healthy"
    
    metrics = {"val_loss": 0.15, "val_accuracy": 95.5}
    meta.set_training_metrics(metrics)
    assert meta.metadata["training"]["metrics"]["val_accuracy"] == 95.5

def test_metadata_exports(tmp_path, sample_config):
    """Test exporting metadata.json, labels.json, and labels.txt to disk."""
    meta = ModelMetadata(sample_config)
    label_map = {0: "ClassA", 1: "ClassB"}
    
    output_dir = str(tmp_path)
    paths = meta.export_all(output_dir, label_map)
    
    assert os.path.exists(paths["metadata"])
    assert os.path.exists(paths["labels_json"])
    assert os.path.exists(paths["labels_txt"])
    
    # Load and verify JSON structure
    with open(paths["metadata"], "r") as f:
        data = json.load(f)
        assert data["model"]["architecture"] == "efficientnet_b2"
        assert data["labels"]["0"] == "ClassA"
        
    with open(paths["labels_txt"], "r") as f:
        lines = [line.strip() for line in f.readlines()]
        assert lines == ["ClassA", "ClassB"]

def test_dataset_hash_reproducibility(tmp_path):
    """Test that dataset hashing is quick, robust, and reproducible."""
    # Create two identical directories
    dir1 = os.path.join(tmp_path, "dataset1")
    dir2 = os.path.join(tmp_path, "dataset2")
    os.makedirs(dir1, exist_ok=True)
    os.makedirs(dir2, exist_ok=True)
    
    for d in [dir1, dir2]:
        os.makedirs(os.path.join(d, "Tomato"), exist_ok=True)
        with open(os.path.join(d, "Tomato", "img1.txt"), "w") as f:
            f.write("leaf content")
            
    hash1 = ModelMetadata.calculate_dataset_hash(dir1)
    hash2 = ModelMetadata.calculate_dataset_hash(dir2)
    
    assert hash1 != "none"
    assert hash1 == hash2, "Identical directories did not produce the same hash"
    
    # Modify one directory (add a file) and check that hash changes
    with open(os.path.join(dir2, "Tomato", "img2.txt"), "w") as f:
        f.write("new file")
        
    hash3 = ModelMetadata.calculate_dataset_hash(dir2)
    assert hash1 != hash3, "Modifying files did not change the hash"
