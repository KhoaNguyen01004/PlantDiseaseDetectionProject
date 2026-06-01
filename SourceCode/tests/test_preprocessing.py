import os
import pytest
from src.train import build_dataloaders

def test_build_dataloaders(mock_dataset_root, monkeypatch):
    """Test building dataloaders with standard splits and customized batch size."""
    # Mock os.path.exists to return False for "data/unknown" to isolate test
    import os
    orig_exists = os.path.exists
    def mock_exists(path):
        if "data/unknown" in path.replace("\\", "/"):
            return False
        return orig_exists(path)
    monkeypatch.setattr(os.path, "exists", mock_exists)

    config = {
        "data": {
            "train_split": 0.5,
            "val_split": 0.25,
            "test_split": 0.25,
            "seed": 42
        },
        "training": {
            "batch_size": 1
        },
        "augmentation": {
            "grayscale_prob": 0.0,
            "random_erasing_prob": 0.0
        }
    }
    
    # We have 4 mock images in total (2 classes, 2 images per class)
    # With splits: train = 2, val = 1, test = 1
    dataloaders, class_names = build_dataloaders(
        dataset_root=mock_dataset_root,
        batch_size=1,
        image_size=260,
        config=config
    )
    
    assert "train" in dataloaders
    assert "val" in dataloaders
    assert "test" in dataloaders
    
    assert len(class_names) == 2
    assert "Apple___Apple_scab" in class_names
    assert "Tomato___healthy" in class_names
    
    # Verify split sizes
    assert len(dataloaders["train"].dataset) == 2
    assert len(dataloaders["val"].dataset) == 1
    assert len(dataloaders["test"].dataset) == 1

def test_split_reproducibility(mock_dataset_root, monkeypatch):
    """Test that split reproducibility seeds result in identical splits across runs."""
    # Mock os.path.exists to return False for "data/unknown" to isolate test
    import os
    orig_exists = os.path.exists
    def mock_exists(path):
        if "data/unknown" in path.replace("\\", "/"):
            return False
        return orig_exists(path)
    monkeypatch.setattr(os.path, "exists", mock_exists)

    config = {
        "data": {
            "train_split": 0.5,
            "val_split": 0.25,
            "test_split": 0.25,
            "seed": 99
        },
        "training": {
            "batch_size": 1
        }
    }
    
    # Run 1
    dataloaders1, _ = build_dataloaders(
        dataset_root=mock_dataset_root,
        batch_size=1,
        image_size=260,
        config=config
    )
    train_dataset1 = dataloaders1["train"].dataset
    train_samples1 = [train_dataset1[i] for i in range(len(train_dataset1))]
    
    # Run 2
    dataloaders2, _ = build_dataloaders(
        dataset_root=mock_dataset_root,
        batch_size=1,
        image_size=260,
        config=config
    )
    train_dataset2 = dataloaders2["train"].dataset
    train_samples2 = [train_dataset2[i] for i in range(len(train_dataset2))]
    
    # Compare split sizes and samples to verify reproducibility
    assert len(train_samples1) == len(train_samples2), "Train split size differs between runs"
    
    # Compare image tensors and labels to ensure same samples
    for sample1, sample2 in zip(train_samples1, train_samples2):
        assert (sample1[0] == sample2[0]).all(), "Image tensors differ between runs"
        assert sample1[1] == sample2[1], "Labels differ between runs"
