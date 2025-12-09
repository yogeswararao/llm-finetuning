"""
Data Loading utilities for loading and preparing datasets for fine-tuning.
"""
import torch
from torch.utils.data import Dataset, Subset

from datasets import load_dataset


class ImdbDataset(Dataset):
    """Dataset for IMDB movie review classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Initialize IMDB dataset.
        
        Args:
            texts: List or array of text strings
            labels: List or array of labels
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_imdb_data(tokenizer, max_length=512, train_samples=None, 
                   test_samples=None, val_samples=200):
    """
    Load IMDB movie review dataset for sentiment analysis.
    
    Args:
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        train_samples: Number of training samples to use
        test_samples: Number of test samples to use
        val_samples: Number of validation samples to use
        
    Returns:
        Tuple of (train_dataset, test_dataset, val_dataset)
    """
    # Load training data
    train_raw = load_dataset('imdb', split='train')
    if train_samples:
        train_raw = train_raw.select(range(min(train_samples, len(train_raw))))
    
    train_dataset = ImdbDataset(
        train_raw['text'],
        train_raw['label'],
        tokenizer,
        max_length
    )
    
    # Load test data
    test_raw = load_dataset('imdb', split='test')
    if test_samples:
        test_raw = test_raw.select(range(min(test_samples, len(test_raw))))
    
    test_dataset = ImdbDataset(
        test_raw['text'],
        test_raw['label'],
        tokenizer,
        max_length
    )
    
    # Create validation set from test set
    n_test_samples = len(test_dataset)
    first_half = list(range(min(val_samples // 2, n_test_samples)))
    last_half = list(range(
        max(0, n_test_samples - val_samples // 2),
        n_test_samples
    ))
    validation_indices = first_half + last_half
    val_dataset = Subset(test_dataset, validation_indices)
    
    return train_dataset, test_dataset, val_dataset
