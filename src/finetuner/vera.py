"""
VeRA (Vector-based Random Matrix Adaptation) Fine-tuning

VeRA is a variant of LoRA that uses shared random adapter matrices across all
tasks and learns only task-specific vectors. This is extremely parameter-efficient. 
Unlike standard LoRA where both A and B matrices are trainable, VeRA freezes these 
adapter matrices and only trains two small scaling vectors (b and d) for each LoRA module.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import VeraConfig, get_peft_model, TaskType
from src.core_finetuner import CoreFineTuner
from src.utils.data_loader import load_imdb_data


class VeRAFineTuner(CoreFineTuner):
    """
    VeRA fine-tuner implementation using PEFT's native VeRAConfig.
    
    VeRA (Vector-based Random Matrix Adaptation) implements:
    1. Freezes A and B matrices (shared random matrices, vera_A and vera_B)
    2. Each module gets only two trainable vectors (d and b)
    3. Forward pass: W' = W + (b . B) @ (d . A)
    4. Uses much higher rank (e.g., r=256) because only two vectors are trainable.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2,
                 r=256, save_projection=True, projection_prng_key=0,
                 vera_dropout=0.0, d_initial=0.1):
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Configure VeRA using PEFT's native VeRAConfig
        vera_config = VeraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,                      # Rank (default 256, higher than LoRA)
            target_modules=['q_lin', 'v_lin', 'k_lin', 'out_lin'],
            save_projection=save_projection, 
            projection_prng_key=projection_prng_key, 
            vera_dropout=vera_dropout,
            d_initial=d_initial
        )
        
        # Apply VeRA
        model = get_peft_model(base_model, vera_config)
        
        # Initialize base class
        super().__init__(model, tokenizer)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
    
    def run(self, save_model=False):
        print("=" * 60)
        print("VeRA Fine-tuning")
        print("=" * 60)
        
        print("\nLoading IMDB dataset...")
        train_dataset, test_dataset, val_dataset = load_imdb_data(
            tokenizer=self.tokenizer
        )
        
        # Train using base class method
        # Uses higher learning rate because only two scaling vectors per module are trainable (d and b)
        # A and B matrices are frozen (shared random matrices)
        print("\nStarting VeRA training...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=1e-3 
        )
        
        # Save model
        if save_model:
            self.save_model('vera')
        
        # Evaluate the model
        print("\nEvaluating the model...")
        self.evaluate(test_dataset)

