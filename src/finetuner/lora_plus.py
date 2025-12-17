"""
LoRA+ (LoRA with Different Learning Rates) Fine-tuning

LoRA+ is a variant of LoRA that uses different learning rates for the A and B
matrices. Typically, the B matrix uses a much higher learning rate (16-32x)
than the A matrix, which improves training stability and final performance
compared to standard LoRA.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from src.core_finetuner import CoreFineTuner
from src.utils.data_loader import load_imdb_data


class LoRAPlusFineTuner(CoreFineTuner):
    """
    LoRA+ fine-tuner implementation.
    
    Applies LoRA with different learning rates for A and B matrices.
    The B matrix typically uses 16-32x higher learning rate than A.
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
        
        # Initialize base class
        super().__init__(model, tokenizer)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def _get_optimizer(self, learning_rate, **kwargs):
        """
        Get optimizer with different learning rates for A and B matrices.
        
        Args:
            learning_rate: Base learning rate (used as fallback for defaults)
            **kwargs: Can include lr_A and lr_B for explicit learning rates.
                     If not provided, defaults are computed from learning_rate.
        
        Returns:
            Optimizer instance with separate parameter groups
        """
        # Use provided values if available, otherwise compute defaults
        # Defaults: lr_A is half of base, lr_B is 8x base (16x lr_A)
        lr_A = kwargs.get('lr_A', learning_rate / 2)
        lr_B = kwargs.get('lr_B', learning_rate * 8)
        
        # Separate parameters: LoRA A and LoRA B
        lora_A_params = []
        lora_B_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'lora_A' in name:
                    lora_A_params.append(param)
                elif 'lora_B' in name:
                    lora_B_params.append(param)
                else:
                    # Other trainable params (e.g., classifier head)
                    lora_B_params.append(param)  # Use higher LR for other params
        
        # Use different learning rates for A and B
        return AdamW([
            {'params': lora_A_params, 'lr': lr_A},
            {'params': lora_B_params, 'lr': lr_B}
        ])
    
    def run(self, save_model=False):
        print("=" * 60)
        print("LoRA+ Fine-tuning")
        print("=" * 60)
        
        print("\nLoading IMDB dataset...")
        train_dataset, test_dataset, val_dataset = load_imdb_data(
            tokenizer=self.tokenizer
        )
        
        # Train using base class method with custom learning rates
        # Note: LoRA+ uses different learning rates for A and B matrices
        # lr_B is typically 16-32x higher than lr_A
        print("\nStarting LoRA+ training...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=2e-4,  # Base LR
            optimizer_kwargs={
                'lr_A': 1e-4,    # Lower LR for A matrices (learning_rate / 2)
                'lr_B': 16e-4    # Higher LR for B matrices (8x base, 16x lr_A)
            }
        )
        
        # Save model
        if save_model:
            self.save_model('lora_plus')
        
        # Evaluate after training
        print("\nEvaluating the model...")
        self.evaluate(test_dataset)

