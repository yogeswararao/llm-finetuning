"""
AdaLoRA (Adaptive LoRA) Fine-tuning

AdaLoRA is a variant of LoRA that adaptively allocates rank budget to different
modules. Instead of using a fixed rank for all modules like standard LoRA,
AdaLoRA dynamically adjusts the rank for each module based on its importance,
allocating more capacity to important modules and less to less important ones.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import AdaLoraConfig, get_peft_model, TaskType
from src.base_finetuner import BaseFineTuner
from src.utils.data_loader import load_imdb_data


class AdaLoRAFineTuner(BaseFineTuner):
    """
    AdaLoRA fine-tuner implementation.
    
    Applies adaptive rank allocation to LoRA adapters, starting with a higher
    initial rank and adaptively redistributing the rank budget during training
    to focus capacity on the most important modules.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2,
                 target_r=8, init_r=12, lora_alpha=16, lora_dropout=0.1, total_step=1000):
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Configure AdaLoRA
        # Note: total_step is required but will be updated with actual value in train() method
        adalora_config = AdaLoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_r=target_r,  # Target rank budget
            init_r=init_r,      # Initial rank (higher than target)
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['q_lin', 'v_lin', 'k_lin', 'out_lin'],
            bias='none',
            total_step=total_step  # Required for AdaLoRA (default 1000, updated in train())
        )
        
        # Apply AdaLoRA
        model = get_peft_model(base_model, adalora_config)
        
        # Initialize base class
        super().__init__(model, tokenizer)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def train(self, train_dataset, val_dataset, epochs=3, batch_size=16,
              learning_rate=2e-4, gradient_clip_norm=1.0, warmup_ratio=0.1,
              optimizer_kwargs=None):
        """
        Train the model with AdaLoRA.
        AdaLoRA requires total_step to be set for adaptive rank allocation.
        """
        from torch.utils.data import DataLoader
        
        # Setup data loaders to calculate total steps
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Calculate total steps for AdaLoRA
        total_steps = len(train_loader) * epochs
        
        # Set total_step on the PEFT config (required for AdaLoRA)
        if hasattr(self.model, 'peft_config'):
            for adapter_name in self.model.peft_config:
                self.model.peft_config[adapter_name].total_step = total_steps
        
        # Call parent train method
        super().train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_clip_norm=gradient_clip_norm,
            warmup_ratio=warmup_ratio,
            optimizer_kwargs=optimizer_kwargs
        )
    
    def run(self, save_model=False):
        print("=" * 60)
        print("AdaLoRA Fine-tuning")
        print("=" * 60)
        
        print("\nLoading IMDB dataset...")
        train_dataset, test_dataset, val_dataset = load_imdb_data(
            tokenizer=self.tokenizer
        )
        
        # Train using base class method
        # Note: AdaLoRA adaptively allocates rank budget during training
        # Starting with init_r and converging to target_r
        print("\nStarting AdaLoRA training...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=2e-4
        )
        
        # Save model
        if save_model:
            self.save_model('adalora')
        
        # Evaluate after training
        print("\nEvaluating after fine-tuning...")
        self.evaluate(test_dataset)
        
        print("\nAdaLoRA fine-tuning completed!")

