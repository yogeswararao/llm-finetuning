"""
P-Tuning Fine-tuning

P-Tuning is a prompt tuning method that uses a prompt encoder (MLP) to generate
prompt representations. Unlike simple prompt tuning which uses learnable embeddings
directly, P-Tuning uses an MLP encoder to generate prompt representations that
are optimized across transformer layers.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PromptEncoderConfig, get_peft_model, TaskType
from src.base_finetuner import BaseFineTuner
from src.utils.data_loader import load_imdb_data


class PTuningFineTuner(BaseFineTuner):
    """
    P-Tuning fine-tuner implementation using PEFT's native PromptEncoderConfig.
    
    P-Tuning implements:
    1. Prompt encoder (MLP) that generates prompt representations
    2. Deep prompt optimization across transformer layers
    3. Virtual tokens prepended to input sequences
    4. Base model weights remain completely frozen
    
    Unlike simple prompt tuning, P-Tuning uses an encoder to generate prompts,
    allowing for more complex prompt representations and better performance.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2,
                 num_virtual_tokens=20, encoder_hidden_size=None, encoder_num_layers=2,
                 encoder_dropout=0.0):
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
        
        # Use model's hidden size if encoder_hidden_size not specified
        if encoder_hidden_size is None:
            encoder_hidden_size = hidden_size
        
        # Configure P-Tuning using PEFT's native PromptEncoderConfig
        # P-Tuning uses a prompt encoder (MLP) to generate prompt representations
        # The encoder takes virtual token embeddings and produces optimized prompts
        # Required parameters:
        # - num_layers: Number of transformer layers in the model
        # - token_dim: Token embedding dimension (usually hidden_size)
        # - num_attention_heads: Number of attention heads in the model
        p_tuning_config = PromptEncoderConfig(
            task_type=TaskType.SEQ_CLS,
            num_virtual_tokens=num_virtual_tokens,  # Number of virtual prompt tokens
            num_layers=num_layers,  # Number of transformer layers (required)
            token_dim=hidden_size,  # Token embedding dimension (required)
            num_attention_heads=num_attention_heads,  # Number of attention heads (required)
            encoder_hidden_size=encoder_hidden_size,  # Hidden size of prompt encoder
            encoder_num_layers=encoder_num_layers,  # Number of layers in prompt encoder MLP
            encoder_dropout=encoder_dropout,  # Dropout for prompt encoder
            encoder_reparameterization_type="MLP"  # Use MLP for reparameterization
        )
        
        # Apply P-Tuning
        model = get_peft_model(base_model, p_tuning_config)
        
        # Initialize base class
        super().__init__(model, tokenizer)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        self.num_virtual_tokens = num_virtual_tokens
    
    def run(self, save_model=False):
        print("=" * 60)
        print("P-Tuning Fine-tuning")
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
        # Note: P-Tuning only trains the prompt encoder, keeping base model frozen
        print("\nStarting P-Tuning training...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=2e-4
        )
        
        # Save model
        if save_model:
            self.save_model('p_tuning')
        
        # Evaluate after training
        print("\nEvaluating after fine-tuning...")
        self.evaluate(test_dataset)
        
        print("\nP-Tuning fine-tuning completed!")

