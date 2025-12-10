"""
QLoRA (Quantized LoRA) Fine-tuning

QLoRA combines 4-bit quantization with LoRA to enable memory-efficient
fine-tuning of large language models. The base model weights are quantized
to 4-bit precision using NF4 quantization, while LoRA adapters remain in
full precision for effective training.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from src.base_finetuner import BaseFineTuner
from src.utils.data_loader import load_imdb_data
import torch
import shutil
import tempfile


class QLoRAFineTuner(BaseFineTuner):
    """
    QLoRA fine-tuner implementation.
    
    Applies 4-bit quantization to the base model and uses LoRA adapters
    for parameter-efficient fine-tuning. Ideal for memory-constrained environments.
    """
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2,
                 r=8, lora_alpha=16, lora_dropout=0.1):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use checkpointing approach to work around quantization issues with encoder-only models
        # 1. First load model in full precision with classification head
        # 2. Save as checkpoint
        # 3. Reload checkpoint with quantization
        # 4. Make sure classifier head is in full precision and trainable 
        checkpoint_dir = tempfile.mkdtemp(prefix="qlora_checkpoint_")
        
        # Load base model + classification head in normal precision
        model_fp = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        
        # Save checkpoint (model + tokenizer)
        model_fp.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Reload checkpoint as quantized model
        base_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir,
            quantization_config=bnb_config
        )
        
        # Clean up checkpoint directory
        shutil.rmtree(checkpoint_dir)
        
        # Prepare for k-bit training
        base_model = prepare_model_for_kbit_training(base_model)
        
        # Configure LoRA
        # Use FEATURE_EXTRACTION instead of SEQ_CLS to avoid PEFT wrapper issues
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['q_lin', 'k_lin', 'v_lin', 'out_lin'],
            bias='none'
        )
        
        model = get_peft_model(base_model, lora_config)
        
        # Make sure classifier is float32 and trainable
        model.classifier.to(torch.float32)
        for p in model.classifier.parameters():
            p.requires_grad_(True)
        
        # Initialize base class
        super().__init__(model, tokenizer)
        
        self.model.print_trainable_parameters()
    
    def run(self, save_model=False):
        print("=" * 60)
        print("QLoRA Fine-tuning")
        print("=" * 60)
        
        print("\nLoading IMDB dataset...")
        train_dataset, test_dataset, val_dataset = load_imdb_data(
            tokenizer=self.tokenizer
        )
        
        print("\nStarting QLoRA training...")
        self.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=3,
            batch_size=16,
            learning_rate=2e-4
        )
        
        # Save model
        if save_model:
            self.save_model('qlora')
        
        print("\nEvaluating after fine-tuning...")
        self.evaluate(test_dataset)
        print("\nQLoRA fine-tuning completed!")

