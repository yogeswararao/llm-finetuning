"""
Delta-LoRA Fine-tuning (Approximation Method)

Delta-LoRA is a variant of LoRA that updates both the LoRA adapters and the base
model weights. Unlike standard LoRA which uses W' = W + BA, Delta-LoRA incorporates
the difference between the product of low-rank matrices A and B in two consecutive
training steps. The weight update is: W' = W + (A^(t+1)B^(t+1) - A^(t)B^(t)),
where the delta (difference) of the low-rank product is added to the base weights.
This allows direct updates to base model weights while maintaining parameter efficiency.

This implementation uses an approximation method that makes base weights trainable
with a much smaller learning rate (typically 100-1000x smaller than LoRA adapters).
This approximates the delta mechanism through gradient-based updates rather than
explicit delta computation.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from src.core_finetuner import CoreFineTuner
from src.utils.data_loader import load_imdb_data


class DeltaLoRAFineTuner(CoreFineTuner):
    """
    Delta-LoRA fine-tuner implementation (approximation method).
    
    This implementation approximates Delta-LoRA by making base weights trainable
    with a much smaller learning rate, rather than explicitly computing the delta.
    
    Approximation approach:
    - Base weights W are made trainable (unlike standard LoRA where they're frozen)
    - LoRA adapters (A and B) use normal learning rate (e.g., 2e-4)
    - Base weights W use a much smaller learning rate (e.g., 1e-6, 200x smaller)
    - Both are updated via standard gradient descent
    
    This approximates the explicit delta mechanism: W' = W + (A^(t+1)B^(t+1) - A^(t)B^(t))
    by allowing base weights to slowly adapt based on the changes in LoRA adapters.
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
        
        # In standard LoRA, base weights are frozen; here we enable them
        # to approximate the delta updates via gradient descent
        for name, param in model.named_parameters():
            if 'lora' not in name.lower() and not param.requires_grad:
                # Make base weights trainable to receive delta updates
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
        # Approximation: base_lr is much smaller (typically 100-1000x smaller)
        # This slow update approximates the explicit delta mechanism
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
        # Note: This approximation method uses different learning rates for LoRA adapters
        # and base weights. Base weights use a much smaller learning rate (typically
        # 100-1000x smaller) to approximate the explicit delta mechanism without
        # computing the delta explicitly.
        print("\nStarting Delta-LoRA training (approximation method)...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=2e-4,  # Default LR (used for LoRA adapters)
            optimizer_kwargs={
                'lora_lr': 2e-4,  # Learning rate for LoRA adapters
                'base_lr': 1e-6   # Much smaller LR for base model weights (200x smaller)
            }
        )
        
        # Save model
        if save_model:
            self.save_model('delta_lora')
        
        # Evaluate after training
        print("\nEvaluating the model...")
        self.evaluate(test_dataset)

