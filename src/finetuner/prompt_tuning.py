"""
Prompt Tuning Fine-tuning

Prompt Tuning is a parameter-efficient fine-tuning method that adds learnable
prompt tokens to the input sequences. Prompt Tuning prepends trainable prompt embeddings
that guide the model's behavior while keeping all base model weights frozen.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PromptTuningConfig, get_peft_model, TaskType
from src.core_finetuner import CoreFineTuner
from src.utils.data_loader import load_imdb_data


class PromptTuningFineTuner(CoreFineTuner):
    """
    Prompt Tuning fine-tuner implementation using PEFT's native PromptTuningConfig.
    
    Prompt Tuning implements:
    1. Learnable prompt tokens prepended to input sequences
    2. Prompt tokens guide model behavior without modifying base weights
    3. Base model weights remain completely frozen
    
    Prompt Tuning uses learnable embeddings directly, making it highly
    parameter-efficient with only the prompt token embeddings being trainable.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2,
                 num_virtual_tokens=20, prompt_tuning_init="RANDOM"):
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Get model configuration for required parameters
        model_config = base_model.config
        hidden_size = model_config.hidden_size
        num_layers = model_config.n_layers
        num_attention_heads = model_config.n_heads
        
        # Configure Prompt Tuning using PEFT's native PromptTuningConfig
        # Prompt Tuning adds learnable prompt tokens that guide model behavior
        # Required parameters:
        # - num_layers: Number of transformer layers in the model
        # - token_dim: Token embedding dimension (usually hidden_size)
        # - num_attention_heads: Number of attention heads in the model
        prompt_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=num_virtual_tokens,  # Number of prompt tokens
            num_layers=num_layers,  # Number of transformer layers (required)
            token_dim=hidden_size,  # Token embedding dimension (required)
            num_attention_heads=num_attention_heads,  # Number of attention heads (required)
            prompt_tuning_init=prompt_tuning_init  # Initialization: "RANDOM" or "TEXT"
        )
        
        # Apply Prompt Tuning
        model = get_peft_model(base_model, prompt_config)
        
        # Initialize base class
        super().__init__(model, tokenizer)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        self.num_virtual_tokens = num_virtual_tokens
    
    def run(self, save_model=False):
        print("=" * 60)
        print("Prompt Tuning Fine-tuning")
        print("=" * 60)
        
        print("\nLoading IMDB dataset...")
        # Reduce max_length to account for virtual tokens
        # Default max_length is 512, so we use 512 - num_virtual_tokens
        max_length = 512 - self.num_virtual_tokens
        train_dataset, test_dataset, val_dataset = load_imdb_data(
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        # Train using base class method
        # Note: Prompt Tuning only trains the prompt tokens, keeping base model frozen
        print("\nStarting Prompt Tuning training...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=2e-4
        )
        
        # Save model
        if save_model:
            self.save_model('prompt_tuning')
        
        # Evaluate after training
        print("\nEvaluating the model...")
        self.evaluate(test_dataset)

