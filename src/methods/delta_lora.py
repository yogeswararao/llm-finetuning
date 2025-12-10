"""
Delta-LoRA Fine-tuning

Delta-LoRA is a variant of LoRA that updates both the LoRA adapters and the base
model weights. Unlike standard LoRA which only trains the low-rank matrices,
Delta-LoRA also makes base model weights trainable with a much smaller learning
rate, allowing direct weight updates while maintaining parameter efficiency.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from src.base_finetuner import BaseFineTuner
from src.utils.data_loader import load_imdb_data


class DeltaLoRAFineTuner(BaseFineTuner):
    """
    Delta-LoRA fine-tuner implementation.
    
    Applies LoRA adapters while also making base model weights trainable with
    a much smaller learning rate. This combines the efficiency of LoRA with
    the ability to directly update base weights.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2,
                 r=8, lora_alpha=16, lora_dropout=0.1):
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['q_lin', 'v_lin', 'k_lin', 'out_lin'],
            bias='none'
        )
        
        # Apply LoRA
        model = get_peft_model(base_model, lora_config)
        
        # Delta-LoRA: Make base model weights trainable
        # In standard LoRA, base weights are frozen; here we enable them
        for name, param in model.named_parameters():
            if 'lora' not in name.lower() and not param.requires_grad:
                # Make base weights trainable for delta updates
                param.requires_grad = True
        
        # Initialize base class
        super().__init__(model, tokenizer)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def _get_optimizer(self, learning_rate, **kwargs):
        """
        Get optimizer with different learning rates for LoRA and base weights.
        
        Args:
            learning_rate: Base learning rate (used for LoRA adapters)
            **kwargs: Can include lora_lr and base_lr for explicit learning rates.
                     If not provided, defaults are computed from learning_rate.
        
        Returns:
            Optimizer instance with separate parameter groups
        """
        # Use provided values if available, otherwise compute defaults
        # Defaults: base_lr is much smaller (typically 100-1000x smaller)
        lora_lr = kwargs.get('lora_lr', learning_rate)
        base_lr = kwargs.get('base_lr', learning_rate / 200)  # 200x smaller by default
        
        # Separate parameters: LoRA adapters and base model weights
        lora_params = []
        base_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora' in name.lower():
                    lora_params.append(param)
                else:
                    base_params.append(param)
        
        # Use different learning rates for LoRA and base weights
        return AdamW([
            {'params': lora_params, 'lr': lora_lr},
            {'params': base_params, 'lr': base_lr}  # Much smaller LR for base
        ])
    
    def run(self, save_model=False):
        print("=" * 60)
        print("Delta-LoRA Fine-tuning")
        print("=" * 60)
        
        print("\nLoading IMDB dataset...")
        train_dataset, test_dataset, val_dataset = load_imdb_data(
            tokenizer=self.tokenizer
        )
        
        # Train using base class method with custom learning rates
        # Note: Delta-LoRA uses different learning rates for LoRA adapters and base weights
        # Base weights use a much smaller learning rate (typically 100-1000x smaller)
        print("\nStarting Delta-LoRA training...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=2e-4,  # Base LR (used for LoRA adapters)
            optimizer_kwargs={
                'lora_lr': 2e-4,  # Learning rate for LoRA adapters
                'base_lr': 1e-6   # Much smaller LR for base model weights (200x smaller)
            }
        )
        
        # Save model
        if save_model:
            self.save_model('delta_lora')
        
        # Evaluate after training
        print("\nEvaluating after fine-tuning...")
        self.evaluate(test_dataset)
        
        print("\nDelta-LoRA fine-tuning completed!")

