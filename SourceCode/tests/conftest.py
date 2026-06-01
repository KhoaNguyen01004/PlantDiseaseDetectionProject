import os
import pytest
import yaml
import numpy as np
import cv2
from PIL import Image

@pytest.fixture
def project_root():
    """Returns the absolute path to the SourceCode project root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture
def config_path(project_root):
    """Returns the path to the centralized configuration file."""
    path = os.path.join(project_root, "configs", "config.yaml")
    if not os.path.exists(path):
        # Fallback
        path = os.path.join(project_root, "SourceCode", "configs", "config.yaml")
    return path

@pytest.fixture
def sample_config(config_path):
    """Loads and returns the actual config dictionary."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def mock_image_numpy():
    """Generates a dummy 3-channel RGB image as a numpy array."""
    # Create a 300x300 RGB image with some patterns to avoid being totally black (useful for validation tests)
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Draw some shapes
    cv2.circle(img, (150, 150), 80, (100, 200, 100), -1)
    cv2.rectangle(img, (50, 50), (100, 100), (50, 50, 150), -1)
    return img

@pytest.fixture
def mock_image_file(tmp_path, mock_image_numpy):
    """Saves a dummy image to a temporary file and returns the path."""
    image_path = os.path.join(tmp_path, "test_leaf.jpg")
    cv2.imwrite(image_path, cv2.cvtColor(mock_image_numpy, cv2.COLOR_RGB2BGR))
    return image_path

@pytest.fixture
def mock_dataset_root(tmp_path):
    """Creates a temporary mock dataset folder structure for dataloader testing."""
    dataset_dir = os.path.join(tmp_path, "mock_data")
    color_dir = os.path.join(dataset_dir, "color")
    os.makedirs(color_dir, exist_ok=True)
    
    # Create 2 mock class directories
    classes = ["Apple___Apple_scab", "Tomato___healthy"]
    for c in classes:
        class_path = os.path.join(color_dir, c)
        os.makedirs(class_path, exist_ok=True)
        # Create a mock image in each class
        img = np.ones((260, 260, 3), dtype=np.uint8) * 128
        cv2.imwrite(os.path.join(class_path, "image1.jpg"), img)
        cv2.imwrite(os.path.join(class_path, "image2.jpg"), img)
        
    return color_dir
