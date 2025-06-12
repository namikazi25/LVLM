#!/usr/bin/env python3
"""Test suite for Phase 3.2 Data Pipeline.

This module tests the data pipeline components including sampling,
augmentation, and pipeline integration.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from mmfakebench.datasets import (
    BaseDataset, MMFakeBenchDataset, MOCHEGDataset,
    BaseSampler, RandomSampler, StratifiedSampler, BalancedSampler, SequentialSampler,
    create_sampler, BaseAugmentation, TextAugmentation, LabelPreservingAugmentation,
    CompositeAugmentation, create_augmentation_pipeline, DataPipeline, create_pipeline_from_config
)


def create_test_data() -> List[Dict[str, Any]]:
    """Create test data for pipeline testing."""
    test_data = [
        {
            "id": "test_1",
            "text": "This is a fake news article about politics",
            "image_path": "test_image_1.jpg",
            "gt_answers": "Fake",
            "fake_cls": "political_fake",
            "text_source": "test",
            "image_source": "test"
        },
        {
            "id": "test_2", 
            "text": "This is a real news article about science",
            "image_path": "test_image_2.jpg",
            "gt_answers": "True",
            "fake_cls": "original",
            "text_source": "test",
            "image_source": "test"
        },
        {
            "id": "test_3",
            "text": "Another fake news story about celebrities",
            "image_path": "test_image_3.jpg", 
            "gt_answers": "Fake",
            "fake_cls": "celebrity_fake",
            "text_source": "test",
            "image_source": "test"
        },
        {
            "id": "test_4",
            "text": "Authentic news about technology advances",
            "image_path": "test_image_4.jpg",
            "gt_answers": "True",
            "fake_cls": "original",
            "text_source": "test",
            "image_source": "test"
        },
        {
            "id": "test_5",
            "text": "Misleading information about health",
            "image_path": "test_image_5.jpg",
            "gt_answers": "Fake",
            "fake_cls": "health_fake",
            "text_source": "test",
            "image_source": "test"
        }
    ]
    return test_data


def test_sampling():
    """Test data sampling functionality."""
    print("\n=== Testing Data Sampling ===")
    
    # Create test data
    test_data = create_test_data()
    
    # Test RandomSampler
    print("Testing RandomSampler...")
    random_sampler = RandomSampler(random_seed=42)
    random_samples = random_sampler.sample(test_data, sample_size=3)
    assert len(random_samples) == 3
    assert all(isinstance(item, dict) for item in random_samples)
    print(f"✅ RandomSampler: sampled {len(random_samples)} items")
    
    # Test SequentialSampler
    print("Testing SequentialSampler...")
    sequential_sampler = SequentialSampler()
    sequential_samples = sequential_sampler.sample(test_data, sample_size=2)
    assert len(sequential_samples) == 2
    assert sequential_samples[0]['id'] == 'test_1'
    assert sequential_samples[1]['id'] == 'test_2'
    print(f"✅ SequentialSampler: sampled {len(sequential_samples)} items")
    
    # Test StratifiedSampler
    print("Testing StratifiedSampler...")
    stratified_sampler = StratifiedSampler(label_key='gt_answers', random_seed=42)
    stratified_samples = stratified_sampler.sample(test_data, sample_size=4)
    assert len(stratified_samples) == 4
    print(f"✅ StratifiedSampler: sampled {len(stratified_samples)} items")
    
    # Test BalancedSampler
    print("Testing BalancedSampler...")
    balanced_sampler = BalancedSampler(label_key='gt_answers', random_seed=42)
    balanced_samples = balanced_sampler.sample(test_data, sample_size=4)
    # Should have equal representation of both classes
    labels = [item['gt_answers'] for item in balanced_samples]
    label_counts = {"True": labels.count("True"), "Fake": labels.count("Fake")}
    assert label_counts["True"] == label_counts["Fake"]
    print(f"✅ BalancedSampler: balanced {len(balanced_samples)} items")
    
    # Test create_sampler factory
    print("Testing create_sampler factory...")
    factory_sampler = create_sampler('random', random_seed=42)
    assert isinstance(factory_sampler, RandomSampler)
    factory_samples = factory_sampler.sample(test_data, sample_size=2)
    assert len(factory_samples) == 2
    print("✅ create_sampler factory works")
    
    print("✅ All sampling tests passed!")


def test_augmentation():
    """Test data augmentation functionality."""
    print("\n=== Testing Data Augmentation ===")
    
    # Test TextAugmentation
    print("Testing TextAugmentation...")
    text_aug = TextAugmentation(augmentation_type='synonym', probability=1.0, random_seed=42)
    test_item = {
        'text': 'This fake news shows real information',
        'gt_answers': 'Fake',
        'fake_cls': 'political_fake'
    }
    
    augmented_item = text_aug.augment_item(test_item)
    assert 'original_text' in augmented_item
    assert 'augmentation_type' in augmented_item
    assert augmented_item['text'] != test_item['text']  # Text should be modified
    print(f"✅ TextAugmentation: '{test_item['text']}' -> '{augmented_item['text']}'")
    
    # Test LabelPreservingAugmentation
    print("Testing LabelPreservingAugmentation...")
    label_preserving_aug = LabelPreservingAugmentation(probability=1.0, random_seed=42)
    preserved_item = label_preserving_aug.augment_item(test_item)
    assert preserved_item['gt_answers'] == test_item['gt_answers']
    assert preserved_item['fake_cls'] == test_item['fake_cls']
    assert preserved_item.get('is_augmented', False)
    print("✅ LabelPreservingAugmentation: labels preserved")
    
    # Test CompositeAugmentation
    print("Testing CompositeAugmentation...")
    text_aug1 = TextAugmentation(augmentation_type='synonym', probability=1.0, random_seed=42)
    text_aug2 = TextAugmentation(augmentation_type='noise', probability=1.0, random_seed=42)
    composite_aug = CompositeAugmentation([text_aug1, text_aug2], probability=1.0, random_seed=42)
    composite_item = composite_aug.augment_item(test_item)
    assert 'applied_augmentations' in composite_item
    print(f"✅ CompositeAugmentation: applied {composite_item.get('applied_augmentations', [])}")
    
    # Test create_augmentation_pipeline factory
    print("Testing create_augmentation_pipeline factory...")
    aug_config = {
        'enabled': True,
        'text_augmentation': {
            'enabled': True,
            'type': 'synonym',
            'probability': 0.5
        },
        'probability': 0.8,
        'random_seed': 42
    }
    factory_aug = create_augmentation_pipeline(aug_config)
    assert factory_aug is not None
    assert isinstance(factory_aug, LabelPreservingAugmentation)
    print("✅ create_augmentation_pipeline factory works")
    
    print("✅ All augmentation tests passed!")


def test_data_pipeline():
    """Test the complete data pipeline."""
    print("\n=== Testing Data Pipeline ===")
    
    # Create temporary test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data file
        test_data_file = temp_path / "test_data.json"
        test_data = create_test_data()
        with open(test_data_file, 'w') as f:
            json.dump(test_data, f)
        
        # Create test image files
        for i in range(1, 6):
            image_file = temp_path / f"test_image_{i}.jpg"
            image_file.write_text("fake image content")  # Placeholder
        
        # Test with MMFakeBenchDataset
        print("Testing pipeline with MMFakeBenchDataset...")
        dataset = MMFakeBenchDataset(data_path=str(test_data_file), image_dir=str(temp_path))
        
        # Create sampler and augmentation
        sampler = RandomSampler(random_seed=42)
        augmentation = TextAugmentation(probability=0.5, random_seed=42)
        
        # Create pipeline
        pipeline = DataPipeline(
            dataset=dataset,
            sampler=sampler,
            sample_size=3,
            augmentation=augmentation,
            random_seed=42
        )
        
        # Test processing
        processed_data = pipeline.process()
        assert len(processed_data) == 3  # Should be sampled to 3 items
        print(f"✅ Pipeline processed {len(processed_data)} items")
        
        # Test statistics
        stats = pipeline.get_statistics()
        assert 'total_items' in stats
        assert 'pipeline_config' in stats
        assert 'labels' in stats
        assert 'text' in stats
        assert 'images' in stats
        assert 'augmentation' in stats
        print(f"✅ Pipeline statistics: {stats['total_items']} items, {stats['labels']['unique_labels']} unique labels")
        
        # Test preview
        preview = pipeline.preview_data(num_samples=2)
        assert len(preview) <= 2
        assert all('text_preview' in item for item in preview)
        print(f"✅ Pipeline preview: {len(preview)} sample items")
        
        # Test iteration
        count = 0
        for item in pipeline:
            count += 1
            assert isinstance(item, dict)
        assert count == len(processed_data)
        print(f"✅ Pipeline iteration: {count} items")
        
        # Test indexing
        first_item = pipeline[0]
        assert isinstance(first_item, dict)
        print("✅ Pipeline indexing works")
        
        # Test export functions
        stats_file = temp_path / "stats.json"
        preview_file = temp_path / "preview.json"
        
        pipeline.export_statistics(stats_file)
        pipeline.export_preview(preview_file, num_samples=2)
        
        assert stats_file.exists()
        assert preview_file.exists()
        print("✅ Pipeline export functions work")
    
    print("✅ All data pipeline tests passed!")


def test_pipeline_factory():
    """Test the pipeline factory function."""
    print("\n=== Testing Pipeline Factory ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test data file
        test_data_file = temp_path / "test_data.json"
        test_data = create_test_data()
        with open(test_data_file, 'w') as f:
            json.dump(test_data, f)
        
        # Create test image files
        for i in range(1, 6):
            image_file = temp_path / f"test_image_{i}.jpg"
            image_file.write_text("fake image content")
        
        # Test pipeline configuration
        config = {
            'dataset': {
                'type': 'mmfakebench',
                'data_path': str(test_data_file),
                'image_dir': str(temp_path)
            },
            'sampling': {
                'enabled': True,
                'type': 'random',
                'sample_size': 3,
                'random_seed': 42
            },
            'augmentation': {
                'enabled': True,
                'text_augmentation': {
                    'enabled': True,
                    'type': 'synonym',
                    'probability': 0.5
                },
                'probability': 0.8,
                'random_seed': 42
            },
            'cache_enabled': True,
            'random_seed': 42
        }
        
        # Create pipeline from config
        pipeline = create_pipeline_from_config(config)
        assert isinstance(pipeline, DataPipeline)
        assert isinstance(pipeline.dataset, MMFakeBenchDataset)
        assert isinstance(pipeline.sampler, RandomSampler)
        assert pipeline.augmentation is not None
        
        # Test processing
        processed_data = pipeline.process()
        assert len(processed_data) == 3
        print(f"✅ Factory pipeline processed {len(processed_data)} items")
        
        # Test with MOCHEG dataset type
        config['dataset']['type'] = 'mocheg'
        mocheg_pipeline = create_pipeline_from_config(config)
        assert isinstance(mocheg_pipeline.dataset, MOCHEGDataset)
        print("✅ Factory supports MOCHEG dataset")
    
    print("✅ All pipeline factory tests passed!")


def test_error_handling():
    """Test error handling in pipeline components."""
    print("\n=== Testing Error Handling ===")
    
    # Test invalid sampler configuration
    try:
        create_sampler('invalid_sampler')
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Invalid sampler type properly handled")
    
    # Test invalid augmentation configuration
    try:
        TextAugmentation(augmentation_type='invalid_type')
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Invalid augmentation type properly handled")
    
    # Test invalid probability values
    try:
        TextAugmentation(probability=1.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Invalid probability properly handled")
    
    # Test invalid dataset type in factory
    try:
        config = {
            'dataset': {
                'type': 'invalid_dataset',
                'data_path': 'dummy_path'
            }
        }
        create_pipeline_from_config(config)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✅ Invalid dataset type properly handled")
    
    print("✅ All error handling tests passed!")


def main():
    """Run all Phase 3.2 tests."""
    print("🧪 Running Phase 3.2 Data Pipeline Tests")
    print("=" * 50)
    
    try:
        test_sampling()
        test_augmentation()
        test_data_pipeline()
        test_pipeline_factory()
        test_error_handling()
        
        print("\n" + "=" * 50)
        print("🎉 All Phase 3.2 tests passed successfully!")
        print("✅ Data sampling implemented and tested")
        print("✅ Data augmentation implemented and tested")
        print("✅ Data pipeline implemented and tested")
        print("✅ Statistics and preview functions working")
        print("✅ Error handling properly implemented")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())