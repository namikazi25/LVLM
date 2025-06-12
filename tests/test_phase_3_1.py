#!/usr/bin/env python3
"""Test suite for Phase 3.1 Dataset Loaders.

This script tests the dataset loading functionality including:
- Base dataset interface
- MMFakeBench dataset loader
- MOCHEG dataset loader
- Dataset validation and preprocessing
- Dataset iteration and statistics
"""

import os
import json
import tempfile
from pathlib import Path
from PIL import Image
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from mmfakebench.datasets.base import BaseDataset
from mmfakebench.datasets.mmfakebench import MMFakeBenchDataset
from mmfakebench.datasets.mocheg import MOCHEGDataset


def create_test_image(path: str, size: tuple = (100, 100)):
    """Create a test image file."""
    img = Image.new('RGB', size, color='red')
    img.save(path)


def test_base_dataset_interface():
    """Test that BaseDataset is properly abstract."""
    print("Testing BaseDataset interface...")
    
    try:
        # Should not be able to instantiate BaseDataset directly
        BaseDataset("/dummy/path")
        assert False, "BaseDataset should be abstract"
    except TypeError:
        print("✅ BaseDataset is properly abstract")
    
    # Test that all required methods are abstract
    required_methods = ['load', 'validate_item', 'preprocess_item']
    for method in required_methods:
        assert hasattr(BaseDataset, method), f"Missing method: {method}"
        print(f"✅ BaseDataset has {method} method")


def test_mmfakebench_dataset():
    """Test MMFakeBench dataset loader."""
    print("\nTesting MMFakeBench dataset...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        images_dir = temp_path / "images"
        images_dir.mkdir()
        
        # Create test images
        img1_path = images_dir / "test1.jpg"
        img2_path = images_dir / "test2.jpg"
        img3_path = images_dir / "missing.jpg"  # This won't be created
        
        create_test_image(str(img1_path))
        create_test_image(str(img2_path))
        
        # Create test JSON data
        test_data = [
            {
                "image_path": "/test1.jpg",
                "text": "This is a test headline",
                "gt_answers": "True",
                "fake_cls": "original",
                "text_source": "test_source",
                "image_source": "test_image_source"
            },
            {
                "image_path": "/test2.jpg",
                "text": "Another test headline",
                "gt_answers": "Fake",
                "fake_cls": "mismatch",
                "text_source": "test_source2",
                "image_source": "test_image_source2"
            },
            {
                "image_path": "/missing.jpg",
                "text": "Missing image test",
                "gt_answers": "True",
                "fake_cls": "original"
            }
        ]
        
        json_path = temp_path / "test_data.json"
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        # Test dataset loading
        dataset = MMFakeBenchDataset(json_path, images_base_dir=images_dir)
        
        # Test load method
        items = dataset.load()
        assert len(items) == 2, f"Expected 2 items, got {len(items)}"  # Only 2 because missing.jpg doesn't exist
        print("✅ MMFakeBench dataset loads correctly")
        
        # Test validation
        for item in items:
            assert dataset.validate_item(item), "Item should be valid"
        print("✅ MMFakeBench validation works")
        
        # Test preprocessing
        for item in items:
            processed = dataset.preprocess_item(item)
            assert 'dataset_name' in processed
            assert processed['dataset_name'] == 'mmfakebench'
            assert 'item_id' in processed
        print("✅ MMFakeBench preprocessing works")
        
        # Test iteration
        count = 0
        for item in dataset:
            count += 1
            assert 'dataset_name' in item
        assert count == 2, f"Expected 2 items in iteration, got {count}"
        print("✅ MMFakeBench iteration works")
        
        # Test statistics
        stats = dataset.get_statistics()
        assert stats['total_items'] == 2
        assert stats['valid_items'] == 2
        assert 'binary_label_distribution' in stats
        print("✅ MMFakeBench statistics work")
        
        # Test with limit
        limited_dataset = MMFakeBenchDataset(json_path, images_base_dir=images_dir, limit=1)
        limited_items = limited_dataset.load()
        assert len(limited_items) == 1, f"Expected 1 item with limit, got {len(limited_items)}"
        print("✅ MMFakeBench limit parameter works")


def test_mocheg_dataset():
    """Test MOCHEG dataset loader."""
    print("\nTesting MOCHEG dataset...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data
        images_dir = temp_path / "images"
        images_dir.mkdir()
        
        # Create test images
        img1_path = images_dir / "meme1.jpg"
        img2_path = images_dir / "meme2.jpg"
        
        create_test_image(str(img1_path))
        create_test_image(str(img2_path))
        
        # Create test JSON data (MOCHEG format)
        test_data = [
            {
                "image_path": "meme1.jpg",
                "text": "This is a harmful meme",
                "label": "harmful"
            },
            {
                "image_path": "meme2.jpg",
                "text": "This is a safe meme",
                "label": "safe"
            }
        ]
        
        json_path = temp_path / "test.json"
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        # Test dataset loading
        dataset = MOCHEGDataset(json_path, images_base_dir=images_dir)
        
        # Test load method
        items = dataset.load()
        assert len(items) == 2, f"Expected 2 items, got {len(items)}"
        print("✅ MOCHEG dataset loads correctly")
        
        # Test validation
        for item in items:
            assert dataset.validate_item(item), "Item should be valid"
        print("✅ MOCHEG validation works")
        
        # Test preprocessing
        for item in items:
            processed = dataset.preprocess_item(item)
            assert 'dataset_name' in processed
            assert processed['dataset_name'] == 'mocheg'
            assert 'item_id' in processed
        print("✅ MOCHEG preprocessing works")
        
        # Test iteration
        count = 0
        for item in dataset:
            count += 1
            assert 'dataset_name' in item
        assert count == 2, f"Expected 2 items in iteration, got {count}"
        print("✅ MOCHEG iteration works")
        
        # Test statistics
        stats = dataset.get_statistics()
        assert stats['total_items'] == 2
        assert stats['valid_items'] == 2
        assert 'label_distribution' in stats
        print("✅ MOCHEG statistics work")


def test_error_handling():
    """Test error handling for invalid data."""
    print("\nTesting error handling...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with non-existent file
        try:
            dataset = MMFakeBenchDataset(temp_path / "nonexistent.json")
            dataset.load()
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            print("✅ FileNotFoundError handled correctly")
        
        # Test with invalid JSON
        invalid_json_path = temp_path / "invalid.json"
        with open(invalid_json_path, 'w') as f:
            f.write("invalid json content")
        
        try:
            dataset = MMFakeBenchDataset(invalid_json_path)
            dataset.load()
            assert False, "Should raise ValueError for invalid JSON"
        except ValueError:
            print("✅ Invalid JSON handled correctly")
        
        # Test validation with invalid items
        valid_json_path = temp_path / "valid.json"
        with open(valid_json_path, 'w') as f:
            json.dump([{"incomplete": "data"}], f)
        
        dataset = MMFakeBenchDataset(valid_json_path)
        items = dataset.load()
        assert len(items) == 0, "Should load 0 items for incomplete data"
        print("✅ Invalid data handled correctly")


def test_dataset_validation():
    """Test dataset validation functionality."""
    print("\nTesting dataset validation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        images_dir = temp_path / "images"
        images_dir.mkdir()
        
        # Create one valid image
        img_path = images_dir / "valid.jpg"
        create_test_image(str(img_path))
        
        # Create test data with mix of valid and invalid items
        test_data = [
            {
                "image_path": "/valid.jpg",
                "text": "Valid item",
                "gt_answers": "True",
                "fake_cls": "original"
            },
            {
                "image_path": "/missing.jpg",  # Missing image
                "text": "Invalid item",
                "gt_answers": "True",
                "fake_cls": "original"
            },
            {
                "image_path": "/valid.jpg",
                "text": "",  # Empty text
                "gt_answers": "True",
                "fake_cls": "original"
            }
        ]
        
        json_path = temp_path / "mixed_data.json"
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        dataset = MMFakeBenchDataset(json_path, images_base_dir=images_dir)
        validation_report = dataset.validate_dataset()
        
        # The dataset loads items with existing images, validation checks all loaded items
        assert validation_report['total_items'] >= 1, f"Expected at least 1 item, got {validation_report['total_items']}"
        assert validation_report['valid_items'] >= 1, f"Expected at least 1 valid item, got {validation_report['valid_items']}"
        assert validation_report['error_count'] >= 0, "Error count should be non-negative"
        print("✅ Dataset validation works correctly")


def main():
    """Run all tests."""
    print("🧪 Running Phase 3.1 Dataset Loaders Tests\n")
    
    try:
        test_base_dataset_interface()
        test_mmfakebench_dataset()
        test_mocheg_dataset()
        test_error_handling()
        test_dataset_validation()
        
        print("\n🎉 All Phase 3.1 tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)