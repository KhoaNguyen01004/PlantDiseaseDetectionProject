import os
import pytest
import numpy as np
from src.inference import preprocess_image, load_labels

def test_preprocess_image_float(mock_image_numpy):
    """Test image preprocessing for float32 model input."""
    preprocessed = preprocess_image(mock_image_numpy, image_size=260, input_dtype=np.float32)
    
    # Expected shape: (1, 3, 260, 260) -> (batch, channels, height, width)
    assert preprocessed.shape == (1, 3, 260, 260)
    assert preprocessed.dtype == np.float32
    
    # Check that it's normalized (not all 0s or all 1s or 0-255)
    # The max/min values should lie within a realistic z-score range
    assert -3.0 < np.min(preprocessed) < 3.0
    assert -3.0 < np.max(preprocessed) < 3.0

def test_preprocess_image_uint8(mock_image_numpy):
    """Test image preprocessing for uint8 (quantized) model input."""
    preprocessed = preprocess_image(mock_image_numpy, image_size=260, input_dtype=np.uint8)
    
    assert preprocessed.shape == (1, 3, 260, 260)
    assert preprocessed.dtype == np.uint8
    assert np.min(preprocessed) >= 0
    assert np.max(preprocessed) <= 255

def test_preprocess_image_invalid_path():
    """Test preprocess_image raises ValueError on invalid image path."""
    with pytest.raises(ValueError):
        preprocess_image("nonexistent_image.jpg")

def test_load_labels(tmp_path):
    """Test loading labels from JSON file."""
    labels_file = os.path.join(tmp_path, "mock_labels.json")
    import json
    mock_labels = {"0": "Class1", "1": "Class2"}
    with open(labels_file, "w") as f:
        json.dump(mock_labels, f)
        
    loaded = load_labels(labels_file)
    assert loaded == mock_labels
    assert loaded["0"] == "Class1"
