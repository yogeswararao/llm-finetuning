"""
LoRA-FA (LoRA with Frozen-A) Fine-tuning

LoRA-FA is a variant of LoRA where only the B matrix is trainable while
the A matrix remains frozen. This further reduces trainable parameters
by 50% compared to standard LoRA while maintaining competitive performance.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType
from src.core_finetuner import CoreFineTuner
from src.utils.data_loader import load_imdb_data


class LoRAFAFineTuner(CoreFineTuner):
    """
    LoRA-FA fine-tuner implementation.
    
    Applies LoRA with frozen A matrices, training only the B matrices.
    This reduces trainable parameters while maintaining model performance.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2,
                 r=8, lora_alpha=16, lora_dropout=0.1):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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
        
        model = get_peft_model(base_model, lora_config)
        super().__init__(model, tokenizer)
        
        # Freeze LoRA A matrices, keep B trainable
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                param.requires_grad = False
            elif 'lora_B' in name:
                param.requires_grad = True
        
        self.model.print_trainable_parameters()
    
    def run(self, save_model=False):
        print("=" * 60)
        print("LoRA-FA Fine-tuning")
        print("=" * 60)
        print("Key Feature: Only matrix B is trained, matrix A is frozen")
        
        print("\nLoading IMDB dataset...")
        train_dataset, test_dataset, val_dataset = load_imdb_data(
            tokenizer=self.tokenizer
        )
        
        print("\nStarting LoRA-FA training...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=2e-4
        )
        
        # Save model
        if save_model:
            self.save_model('lora_fa')
        
        print("\nEvaluating the model...")
        self.evaluate(test_dataset)

