"""Data augmentation utilities for MMFakeBench datasets.

This module provides data augmentation techniques for multimodal
misinformation detection datasets.
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import logging
import re


class BaseAugmentation(ABC):
    """Abstract base class for data augmentation."""
    
    def __init__(self, probability: float = 0.5, random_seed: Optional[int] = None):
        """Initialize the augmentation.
        
        Args:
            probability: Probability of applying augmentation to each item
            random_seed: Random seed for reproducible augmentation
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"Probability must be between 0 and 1, got {probability}")
        
        self.probability = probability
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def augment_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Augment a single data item.
        
        Args:
            item: Original data item
            
        Returns:
            Augmented data item
        """
        raise NotImplementedError("Subclasses must implement augment_item method")
    
    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation with probability.
        
        Args:
            item: Original data item
            
        Returns:
            Augmented or original data item
        """
        if random.random() < self.probability:
            return self.augment_item(item)
        return item.copy()


class TextAugmentation(BaseAugmentation):
    """Text-based augmentation techniques."""
    
    def __init__(self, augmentation_type: str = 'synonym', probability: float = 0.5, 
                 random_seed: Optional[int] = None):
        """Initialize text augmentation.
        
        Args:
            augmentation_type: Type of text augmentation ('synonym', 'paraphrase', 'noise')
            probability: Probability of applying augmentation
            random_seed: Random seed for reproducible augmentation
        """
        super().__init__(probability, random_seed)
        self.augmentation_type = augmentation_type.lower()
        
        if self.augmentation_type not in ['synonym', 'paraphrase', 'noise']:
            raise ValueError(f"Unsupported augmentation type: {augmentation_type}")
    
    def augment_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Augment text in the data item.
        
        Args:
            item: Original data item
            
        Returns:
            Item with augmented text
        """
        augmented_item = item.copy()
        
        if 'text' in item and item['text']:
            original_text = item['text']
            
            if self.augmentation_type == 'synonym':
                augmented_text = self._synonym_replacement(original_text)
            elif self.augmentation_type == 'paraphrase':
                augmented_text = self._simple_paraphrase(original_text)
            elif self.augmentation_type == 'noise':
                augmented_text = self._add_noise(original_text)
            else:
                augmented_text = original_text
            
            augmented_item['text'] = augmented_text
            augmented_item['original_text'] = original_text
            augmented_item['augmentation_type'] = self.augmentation_type
        
        return augmented_item
    
    def _synonym_replacement(self, text: str) -> str:
        """Simple synonym replacement (placeholder implementation)."""
        # Simple word replacements for demonstration
        replacements = {
            'fake': 'false',
            'real': 'authentic',
            'true': 'genuine',
            'news': 'information',
            'image': 'picture',
            'photo': 'photograph',
            'says': 'claims',
            'shows': 'displays'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in replacements and random.random() < 0.3:
                # Preserve original case
                if word.isupper():
                    words[i] = replacements[word_lower].upper()
                elif word.istitle():
                    words[i] = replacements[word_lower].title()
                else:
                    words[i] = replacements[word_lower]
        
        return ' '.join(words)
    
    def _simple_paraphrase(self, text: str) -> str:
        """Simple paraphrasing (placeholder implementation)."""
        # Simple sentence structure changes
        paraphrases = [
            (r'This is (.+)', r'Here we see \1'),
            (r'The (.+) shows (.+)', r'\1 displays \2'),
            (r'(.+) claims (.+)', r'According to \1, \2'),
            (r'(.+) says (.+)', r'\1 states \2')
        ]
        
        augmented_text = text
        for pattern, replacement in paraphrases:
            if random.random() < 0.3:
                augmented_text = re.sub(pattern, replacement, augmented_text, flags=re.IGNORECASE)
        
        return augmented_text
    
    def _add_noise(self, text: str) -> str:
        """Add minor noise to text."""
        words = text.split()
        
        # Randomly swap adjacent words
        if len(words) > 1 and random.random() < 0.2:
            idx = random.randint(0, len(words) - 2)
            words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        # Randomly duplicate a word
        if len(words) > 0 and random.random() < 0.1:
            idx = random.randint(0, len(words) - 1)
            words.insert(idx + 1, words[idx])
        
        return ' '.join(words)


class LabelPreservingAugmentation(BaseAugmentation):
    """Augmentation that preserves label consistency."""
    
    def __init__(self, text_augmentation: Optional[TextAugmentation] = None,
                 probability: float = 0.5, random_seed: Optional[int] = None):
        """Initialize label-preserving augmentation.
        
        Args:
            text_augmentation: Text augmentation to apply
            probability: Probability of applying augmentation
            random_seed: Random seed for reproducible augmentation
        """
        super().__init__(probability, random_seed)
        self.text_augmentation = text_augmentation or TextAugmentation(random_seed=random_seed)
    
    def augment_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Augment item while preserving labels.
        
        Args:
            item: Original data item
            
        Returns:
            Augmented item with preserved labels
        """
        # Apply text augmentation
        augmented_item = self.text_augmentation.augment_item(item)
        
        # Ensure all labels are preserved
        label_keys = ['label_binary', 'label_multiclass', 'label']
        for key in label_keys:
            if key in item:
                augmented_item[key] = item[key]
        
        # Mark as augmented
        augmented_item['is_augmented'] = True
        augmented_item['augmentation_applied'] = True
        
        return augmented_item


class CompositeAugmentation(BaseAugmentation):
    """Composite augmentation that applies multiple augmentations."""
    
    def __init__(self, augmentations: List[BaseAugmentation], 
                 probability: float = 0.5, random_seed: Optional[int] = None):
        """Initialize composite augmentation.
        
        Args:
            augmentations: List of augmentations to apply
            probability: Probability of applying the composite augmentation
            random_seed: Random seed for reproducible augmentation
        """
        super().__init__(probability, random_seed)
        self.augmentations = augmentations
    
    def augment_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple augmentations sequentially.
        
        Args:
            item: Original data item
            
        Returns:
            Item with multiple augmentations applied
        """
        augmented_item = item.copy()
        applied_augmentations = []
        
        for augmentation in self.augmentations:
            if random.random() < augmentation.probability:
                augmented_item = augmentation.augment_item(augmented_item)
                applied_augmentations.append(augmentation.__class__.__name__)
        
        if applied_augmentations:
            augmented_item['applied_augmentations'] = applied_augmentations
        
        return augmented_item


def create_augmentation_pipeline(config: Dict[str, Any]) -> Optional[BaseAugmentation]:
    """Create an augmentation pipeline from configuration.
    
    Args:
        config: Augmentation configuration
        
    Returns:
        Configured augmentation pipeline or None if disabled
    """
    if not config.get('enabled', False):
        return None
    
    augmentations = []
    
    # Text augmentation
    if config.get('text_augmentation', {}).get('enabled', False):
        text_config = config['text_augmentation']
        text_aug = TextAugmentation(
            augmentation_type=text_config.get('type', 'synonym'),
            probability=text_config.get('probability', 0.5),
            random_seed=config.get('random_seed')
        )
        augmentations.append(text_aug)
    
    if not augmentations:
        return None
    
    # Wrap in label-preserving augmentation
    if len(augmentations) == 1:
        return LabelPreservingAugmentation(
            text_augmentation=augmentations[0] if isinstance(augmentations[0], TextAugmentation) else None,
            probability=config.get('probability', 0.5),
            random_seed=config.get('random_seed')
        )
    else:
        composite = CompositeAugmentation(
            augmentations=augmentations,
            probability=config.get('probability', 0.5),
            random_seed=config.get('random_seed')
        )
        return LabelPreservingAugmentation(
            text_augmentation=composite,
            probability=1.0,  # Always apply if composite is selected
            random_seed=config.get('random_seed')
        )