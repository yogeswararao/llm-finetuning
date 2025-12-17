"""
Full Fine-tuning

Full fine-tuning updates all model parameters during training. This approach trains the 
entire model parameters end-to-end, typically achieving the best performance 
but requiring more computational resources and memory.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.core_finetuner import CoreFineTuner
from src.utils.data_loader import load_imdb_data


class FullFineTuner(CoreFineTuner):
    """
    Full fine-tuner implementation.
    
    Fine-tunes all model parameters using standard forward and backward pass.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model (all parameters will be trainable)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
        # Initialize base class
        super().__init__(model, tokenizer)

    def run(self, save_model=False):
        print("=" * 60)
        print("Full Fine-tuning")
        print("=" * 60)

        print("\nLoading IMDB dataset...")
        train_dataset, test_dataset, val_dataset = load_imdb_data(
            tokenizer=self.tokenizer
        )

        # Train using base class method
        # Note: Using 2e-5 learning rate for stable full fine-tuning
        print("\nStarting training...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=2e-5
        )

        # Save model
        if save_model:
            self.save_model('full_finetuner')
    
        # Evaluate after training
        print("\nEvaluating the model...")
        self.evaluate(test_dataset)

