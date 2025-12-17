"""
Evaluation utilities for measuring model performance
"""
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm


def evaluate_model(model, data_loader, device):
    """
    Evaluate model performance on a classification dataset.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation metrics (accuracy, precision, recall, f1_score, loss)
    """
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                total_loss += outputs.loss.item()
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    avg_loss = total_loss / len(data_loader) if total_loss > 0 else None
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'loss': avg_loss,
        'predictions': all_predictions,
        'labels': all_labels
    }
    
    return results


def print_metrics(results):
    """Print evaluation results in a formatted way"""
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    
    if results.get('loss') is not None:
        print(f"Loss: {results['loss']:.4f}")
    
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    
    print(f"{'='*60}\n")
