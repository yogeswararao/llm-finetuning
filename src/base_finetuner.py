"""
Reusable base finetuner class with common training and evaluation methods
"""
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.utils.evaluation import (
    evaluate_model,
    print_metrics
)
from tqdm import tqdm
import os


class BaseFineTuner:
    """
    Base class for fine-tuning with common training and evaluation methods
    
    Subclasses should implement:
    - __init__: Initialize model and tokenizer
    - _get_optimizer: Return optimizer (optional, defaults to AdamW)
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize base fine-tuner
        
        Args:
            model: The model to fine-tune
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _prepare_batch(self, batch):
        """Prepare batch for model input (can be overridden)"""
        return {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'labels': batch['labels'].to(self.device)
        }
    
    def _get_optimizer(self, learning_rate, **kwargs):
        """
        Get optimizer (can be overridden by subclasses)
        
        Args:
            learning_rate: Base learning rate
            **kwargs: Additional optimizer arguments
        
        Returns:
            Optimizer instance
        """
        return AdamW(self.model.parameters(), lr=learning_rate, **kwargs)
    
    def train(self, train_dataset, val_dataset, epochs=3, batch_size=16,
              learning_rate=2e-5, gradient_clip_norm=1.0, warmup_ratio=0.1,
              optimizer_kwargs=None):
        """
        Train the model on the provided datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            gradient_clip_norm: Gradient clipping norm
            warmup_ratio: Ratio of warmup steps
            optimizer_kwargs: Additional optimizer arguments
        
        Returns:
            None
        """
        # Setup data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup optimizer
        opt_kwargs = optimizer_kwargs or {}
        optimizer = self._get_optimizer(learning_rate, **opt_kwargs)
        
        # Setup learning rate scheduler
        total_steps = len(train_loader) * epochs
        num_warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps
        )
        
        # Get trainable parameters for gradient clipping
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Training
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                if gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                
                loss_value = loss.item()
                total_loss += loss_value
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1} - Training Loss: {avg_loss:.4f}')
            
            # Validation
            val_loss = self.validate(val_loader)
            print(f'Epoch {epoch+1} - Validation Loss: {val_loss:.4f}')
    
    def validate(self, val_loader):
        """
        Validation loop
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._prepare_batch(batch)
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.model.train()
        return avg_loss
    
    def evaluate(self, test_dataset, batch_size=32):
        """
        Evaluate model on test dataset
        
        Args:
            test_dataset: Test dataset
            batch_size: Batch size for evaluation
        """
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        results = evaluate_model(
            self.model,
            test_loader,
            self.device
        )
        
        print_metrics(results)
    
    def save_model(self, method_name):
        """
        Save model and tokenizer to disk. Saves to "models" directory.
        
        Args:
            method_name: Method name (e.g., "full_finetuning", "lora") used to construct save path
        """
  
        output_dir = f'models/{method_name}'
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f'Model saved to {output_dir}')
    
    @classmethod
    def load_model(cls, model_path, **model_kwargs):
        """
        Load a saved model and tokenizer from disk.
        
        Args:
            model_path: Path to the saved model directory
            **model_kwargs: Additional arguments to pass to model loading
        
        Returns:
            BaseFineTuner instance with loaded model and tokenizer
        """
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Check if it's a PEFT model and load accordingly
        adapter_config_path = os.path.join(model_path, 'adapter_config.json')
        if os.path.exists(adapter_config_path):
            # PEFT model - get base model name from config and load
            import json
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path')
            
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                **model_kwargs
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Regular model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                **model_kwargs
            )
        
        # Create instance
        instance = cls.__new__(cls)
        instance.model = model
        instance.tokenizer = tokenizer
        instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        instance.model.to(instance.device)
        
        print(f'Model loaded from {model_path}')
        return instance
