"""
LoRA (Low-Rank Adaptation) Fine-tuning

LoRA is a parameter-efficient fine-tuning method that decomposes weight updates
into low-rank matrices. Instead of updating all model parameters like full
fine-tuning, LoRA adds trainable low-rank matrices (A and B) to specific
layers, where the weight update is approximated as ΔW = BA.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from src.core_finetuner import CoreFineTuner
from src.utils.data_loader import load_imdb_data


class LoRAFineTuner(CoreFineTuner):
    """
    LoRA fine-tuner implementation.
    
    Applies Low-Rank Adaptation to transformer layers, specifically targeting
    query, key, value, and output linear layers in the attention mechanism.
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
    
    def run(self, save_model=False):
        print("=" * 60)
        print("LoRA Fine-tuning")
        print("=" * 60)
        
        print("\nLoading IMDB dataset...")
        train_dataset, test_dataset, val_dataset = load_imdb_data(
            tokenizer=self.tokenizer
        )
        
        # Train using base class method
        # Note: Using 2e-4 learning rate (higher than full fine-tuning) since
        # only adapters are trained, allowing for faster learning
        print("\nStarting training...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=2e-4
        )
        
        # Save model
        if save_model:
            self.save_model('lora')
        
        # Evaluate after training
        print("\nEvaluating the model...")
        self.evaluate(test_dataset)
