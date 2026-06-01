import os
import pytest
from src.preprocessing.preprocess import load_config

def test_config_file_exists(config_path):
    """Verify that config.yaml exists at the expected path."""
    assert os.path.exists(config_path), f"Config file not found at: {config_path}"

def test_config_loading(config_path):
    """Verify config loads successfully as a dictionary."""
    config = load_config(config_path)
    assert isinstance(config, dict)
    assert len(config) > 0

def test_config_required_sections(sample_config):
    """Verify that all required configuration sections are present."""
    required_sections = ["model", "image", "training", "data", "augmentation", "quality_validation"]
    for section in required_sections:
        assert section in sample_config, f"Required section '{section}' is missing from config."

def test_config_values_and_types(sample_config):
    """Verify types and bounds of critical config values."""
    # Model
    model = sample_config["model"]
    assert "architecture" in model
    assert isinstance(model["architecture"], str)
    assert model["num_classes"] == 39
    
    # Image
    image = sample_config["image"]
    assert image["size"] == 260
    assert isinstance(image["mean"], list) and len(image["mean"]) == 3
    assert isinstance(image["std"], list) and len(image["std"]) == 3
    
    # Training
    training = sample_config["training"]
    assert training["batch_size"] > 0
    assert training["epochs"] > 0
    assert 0.0 < float(training["learning_rate"]) < 1.0

def test_data_splits_sum_to_one(sample_config):
    """Verify train, val, and test splits sum up to 1.0."""
    data = sample_config["data"]
    total = data["train_split"] + data["val_split"] + data["test_split"]
    assert abs(total - 1.0) < 1e-6, f"Data splits sum up to {total}, which is not 1.0"
