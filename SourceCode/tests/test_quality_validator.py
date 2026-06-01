import pytest
import numpy as np
import cv2
from src.quality_validator import QualityValidator

def test_quality_validator_default_init():
    """Test validator initialization with defaults."""
    validator = QualityValidator()
    assert validator.config["blur_threshold"] == 100.0
    assert validator.config["min_width"] == 224

def test_check_resolution():
    """Test resolution check logic."""
    validator = QualityValidator({
        "min_width": 200, "min_height": 200,
        "max_width": 1000, "max_height": 1000
    })
    
    # Normal
    val, info = validator._check_resolution(500, 500)
    assert val is True
    assert "OK" in info["guidance"]
    
    # Too small
    val, info = validator._check_resolution(100, 500)
    assert val is False
    assert "too low" in info["guidance"]
    
    # Too large (warning but True)
    val, info = validator._check_resolution(1200, 500)
    assert val is True
    assert "too high" in info["guidance"]

def test_check_brightness():
    """Test brightness check logic using dummy solid images."""
    validator = QualityValidator({
        "min_brightness": 50.0,
        "max_brightness": 200.0
    })
    
    # Too dark (0)
    img_dark = np.zeros((100, 100, 3), dtype=np.uint8)
    val, info = validator._check_brightness(img_dark)
    assert val is False
    assert "too dark" in info["guidance"]
    
    # Normal (128)
    img_normal = np.ones((100, 100, 3), dtype=np.uint8) * 128
    val, info = validator._check_brightness(img_normal)
    assert val is True
    assert "Brightness OK" in info["guidance"]
    
    # Too bright (255)
    img_bright = np.ones((100, 100, 3), dtype=np.uint8) * 230
    val, info = validator._check_brightness(img_bright)
    assert val is False
    assert "too bright" in info["guidance"]

def test_check_blur():
    """Test blur check logic."""
    validator = QualityValidator({"blur_threshold": 50.0})
    
    # Generate sharp image (high variance checkerboard or sharp grid)
    sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
    for i in range(200):
        for j in range(200):
            if (i // 10 + j // 10) % 2 == 0:
                sharp_img[i, j] = [255, 255, 255]
                
    val_sharp, info_sharp = validator._check_blur(sharp_img)
    # Checkerboard has extremely high Laplacian variance
    assert val_sharp is True
    assert info_sharp["laplacian_variance"] > 50.0
    
    # Blurry image
    blurry_img = cv2.GaussianBlur(sharp_img, (21, 21), 0)
    val_blur, info_blur = validator._check_blur(blurry_img)
    assert val_blur is False
    assert info_blur["laplacian_variance"] < 50.0

def test_get_user_guidance():
    """Test user guidance string generation."""
    validator = QualityValidator()
    
    success_results = {
        "valid": True,
        "checks": {"resolution": True, "blur": True, "brightness": True},
        "guidance": [],
        "scores": {}
    }
    guidance = validator.get_user_guidance(success_results)
    assert "Ready for classification" in guidance
    
    failure_results = {
        "valid": False,
        "checks": {"resolution": False, "blur": True, "brightness": True},
        "guidance": ["Image resolution too low"],
        "scores": {}
    }
    guidance = validator.get_user_guidance(failure_results)
    assert "issues detected" in guidance
    assert "resolution too low" in guidance
