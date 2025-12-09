"""
VeRA (Vector-based Random Matrix Adaptation) Fine-tuning

VeRA is a variant of LoRA that uses shared random matrices (frozen) across all
tasks and learns only task-specific vectors. This is extremely parameter-efficient,
requiring only ~0.1% of LoRA's parameters while achieving competitive performance.
Unlike standard LoRA where both A and B matrices are trainable, VeRA freezes
these matrices and only trains small scaling vectors (d) for each LoRA module.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import VeraConfig, get_peft_model, TaskType
from src.base_finetuner import BaseFineTuner
from src.utils.data_loader import load_imdb_data


class VeRAFineTuner(BaseFineTuner):
    """
    VeRA fine-tuner implementation using PEFT's native VeRAConfig.
    
    VeRA (Vector-based Random Matrix Adaptation) implements:
    1. Freezing A and B matrices (shared random matrices, vera_A and vera_B)
    2. Training only two scaling vectors per module:
       - lambda_d: size r (scales rank dimension)
       - lambda_b: size out_features (scales output)
    3. Forward pass: W' = W + lambda_b * (lambda_d * (x @ A) @ B)
    
    Each module gets only two trainable vectors (lambda_d + lambda_b),
    making this extremely parameter-efficient (~0.1% of LoRA parameters).
    
    Key difference from LoRA: VeRA can use much higher ranks (default r=256)
    because only two vectors are trainable, not the full A and B matrices.
    This allows higher capacity with similar parameter count.
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
        # VeRA implements: W' = W + lambda_b * (lambda_d * (x @ A) @ B)
        # Where:
        # - A and B are frozen shared random matrices (vera_A, vera_B)
        # - lambda_d is trainable vector (size r) - scales rank dimension
        # - lambda_b is trainable vector (size out_features) - scales output
        # Note: VeRA can use much higher ranks than LoRA because:
        # - Only two vectors are trainable per module (lambda_d + lambda_b)
        # - A and B matrices are frozen (shared random matrices)
        # - Trainable params ≈ r + out_features, not 2 * r * hidden_size like LoRA
        # Default r=256 is recommended for VeRA (vs r=8-32 for LoRA)
        vera_config = VeraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,                      # Rank (default 256, higher than LoRA)
            target_modules=['q_lin', 'v_lin', 'k_lin', 'out_lin'],
            save_projection=save_projection,  # Save projection matrices in checkpoint
            projection_prng_key=projection_prng_key,  # PRNG key (default 0) to regenerate matrices if save_projection=False
            vera_dropout=vera_dropout,  # Dropout probability for VeRA layers (default 0.0)
            d_initial=d_initial  # Initial value for lambda_d vector (default 0.1, small values recommended)
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
        # Note: VeRA uses higher learning rate for scaling vectors (lambda_d, lambda_b)
        # A and B matrices are frozen (shared random matrices)
        # Only two vectors are trainable per module: lambda_d (size r) and lambda_b (size out_features)
        print("\nStarting VeRA training...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=1e-3  # Higher LR for vectors (only trainable params)
        )
        
        # Save model
        if save_model:
            self.save_model('vera')
        
        # Evaluate after training
        print("\nEvaluating after fine-tuning...")
        self.evaluate(test_dataset)
        
        print("\nVeRA fine-tuning completed!")

