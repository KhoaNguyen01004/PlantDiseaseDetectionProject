import os
import pytest
import torch
import torch.nn as nn
from src.evaluate_and_convert import export_onnx

class TinyMockModel(nn.Module):
    """A minimal PyTorch model for fast testing of ONNX export."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Linear(8, 2)
        
    def forward(self, x):
        return self.classifier(self.features(x))

def test_export_onnx(tmp_path):
    """Test that a PyTorch model can be successfully exported to an ONNX file."""
    # Create the mock model
    model = TinyMockModel()
    
    # Save the original working dir
    original_cwd = os.getcwd()
    
    # Move to a temporary directory to avoid writing to the project root during test
    os.chdir(tmp_path)
    
    try:
        # Run export
        onnx_file = export_onnx(model, image_size=260)
        
        # Verify the ONNX file exists
        assert onnx_file == "plant_model.onnx"
        assert os.path.exists(onnx_file)
        assert os.path.getsize(onnx_file) > 0
        
    finally:
        # Restore CWD
        os.chdir(original_cwd)
