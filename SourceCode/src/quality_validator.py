#!/usr/bin/env python3
"""Photo Quality Validation for Plant Disease Detection.

Validates image quality before classification to ensure reliable predictions.
Checks for blur, brightness, and resolution issues.
"""
import cv2
import numpy as np
import argparse
import json
from typing import Tuple, Dict, Optional
from .logging_config import get_logger

logger = get_logger(__name__)


class QualityValidator:
    """Validates image quality for plant disease detection."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize validator with configuration.
        
        Args:
            config: Configuration dictionary with thresholds.
                   If None, uses default thresholds.
        """
        # Default thresholds
        self.config = config or {
            # Blur detection
            "blur_threshold": 100.0,  # Laplacian variance threshold
            
            # Brightness validation
            "min_brightness": 30.0,   # Minimum average brightness (0-255)
            "max_brightness": 220.0,  # Maximum average brightness (0-255)
            
            # Resolution validation
            "min_width": 224,         # Minimum image width
            "min_height": 224,        # Minimum image height
            "max_width": 4000,        # Maximum image width
            "max_height": 4000,       # Maximum image height
        }
    
    def validate(self, image_path: str) -> Tuple[bool, Dict]:
        """Validate image quality.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Tuple of (is_valid, results_dict)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return False, {
                "valid": False,
                "error": f"Cannot load image: {image_path}",
                "guidance": "Please check if the file exists and is a valid image."
            }
        
        return self.validate_from_image(image)
    
    def validate_from_image(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """Validate image quality from loaded image.
        
        Args:
            image: Image as numpy array (H, W, 3) in BGR format
        
        Returns:
            Tuple of (is_valid, results_dict)
        """
        results = {
            "valid": True,
            "checks": {},
            "guidance": [],
            "scores": {}
        }
        
        # Check resolution
        height, width = image.shape[:2]
        resolution_valid, resolution_info = self._check_resolution(width, height)
        results["checks"]["resolution"] = resolution_valid
        results["scores"]["resolution"] = resolution_info
        if not resolution_valid:
            results["valid"] = False
            results["guidance"].append(resolution_info["guidance"])
        
        # Check blur
        blur_valid, blur_info = self._check_blur(image)
        results["checks"]["blur"] = blur_valid
        results["scores"]["blur"] = blur_info
        if not blur_valid:
            results["valid"] = False
            results["guidance"].append(blur_info["guidance"])
        
        # Check brightness
        brightness_valid, brightness_info = self._check_brightness(image)
        results["checks"]["brightness"] = brightness_valid
        results["scores"]["brightness"] = brightness_info
        if not brightness_valid:
            results["valid"] = False
            results["guidance"].append(brightness_info["guidance"])
        
        return results["valid"], results
    
    def _check_resolution(self, width: int, height: int) -> Tuple[bool, Dict]:
        """Check image resolution."""
        info = {
            "width": width,
            "height": height,
            "guidance": ""
        }
        
        if width < self.config["min_width"] or height < self.config["min_height"]:
            info["guidance"] = (
                f"Image resolution too low ({width}x{height}). "
                f"Minimum required: {self.config['min_width']}x{self.config['min_height']}. "
                "Please move closer to the leaf or use a higher resolution camera."
            )
            return False, info
        
        if width > self.config["max_width"] or height > self.config["max_height"]:
            info["guidance"] = (
                f"Image resolution too high ({width}x{height}). "
                f"Maximum allowed: {self.config['max_width']}x{self.config['max_height']}. "
                "Image will be downscaled for processing."
            )
            # Not a failure, just a warning
            return True, info
        
        info["guidance"] = f"Resolution OK ({width}x{height})"
        return True, info
    
    def _check_blur(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """Check image blur using Laplacian variance."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        info = {
            "laplacian_variance": laplacian_var,
            "threshold": self.config["blur_threshold"],
            "guidance": ""
        }
        
        if laplacian_var < self.config["blur_threshold"]:
            info["guidance"] = (
                f"Image is too blurry (sharpness score: {laplacian_var:.1f}, "
                f"threshold: {self.config['blur_threshold']}). "
                "Please hold the camera steady and ensure the leaf is in focus."
            )
            return False, info
        
        info["guidance"] = f"Sharpness OK (score: {laplacian_var:.1f})"
        return True, info
    
    def _check_brightness(self, image: np.ndarray) -> Tuple[bool, Dict]:
        """Check image brightness."""
        # Convert to grayscale and calculate mean brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        info = {
            "mean_brightness": mean_brightness,
            "min_threshold": self.config["min_brightness"],
            "max_threshold": self.config["max_brightness"],
            "guidance": ""
        }
        
        if mean_brightness < self.config["min_brightness"]:
            info["guidance"] = (
                f"Image is too dark (brightness: {mean_brightness:.1f}, "
                f"minimum: {self.config['min_brightness']}). "
                "Please move to a brighter location or use flash."
            )
            return False, info
        
        if mean_brightness > self.config["max_brightness"]:
            info["guidance"] = (
                f"Image is too bright (brightness: {mean_brightness:.1f}, "
                f"maximum: {self.config['max_brightness']}). "
                "Please avoid direct sunlight or move to shade."
            )
            return False, info
        
        info["guidance"] = f"Brightness OK (score: {mean_brightness:.1f})"
        return True, info
    
    def get_user_guidance(self, results: Dict) -> str:
        """Get user-friendly guidance message from validation results.
        
        Args:
            results: Results dictionary from validate()
        
        Returns:
            User-friendly guidance string
        """
        if results["valid"]:
            return "✅ Image quality is good. Ready for classification."
        
        guidance_parts = ["❌ Image quality issues detected:"]
        for guidance in results["guidance"]:
            guidance_parts.append(f"  • {guidance}")
        
        return "\n".join(guidance_parts)


def validate_image(image_path: str, config: Optional[Dict] = None) -> Tuple[bool, str]:
    """Convenience function to validate an image.
    
    Args:
        image_path: Path to image file
        config: Optional configuration dictionary
    
    Returns:
        Tuple of (is_valid, guidance_message)
    """
    validator = QualityValidator(config)
    is_valid, results = validator.validate(image_path)
    guidance = validator.get_user_guidance(results)
    return is_valid, guidance


def main():
    parser = argparse.ArgumentParser(description='Photo Quality Validator for Plant Disease Detection')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--config', default=None, help='Path to config JSON file')
    parser.add_argument('--verbose', action='store_true', help='Show detailed results')
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Validate
    validator = QualityValidator(config)
    is_valid, results = validator.validate(args.image)
    
    # Log results (console user-friendly + structured log)
    logger.info(f"Image: {args.image}")
    if is_valid:
        logger.info("Valid: ✅ YES")
    else:
        logger.warning("Valid: ❌ NO")

    if args.verbose:
        logger.debug("Detailed Results:\n" + json.dumps(results, indent=2))

    logger.info("Guidance: %s", validator.get_user_guidance(results))
    
    # Exit with appropriate code
    exit(0 if is_valid else 1)


if __name__ == '__main__':
    main()