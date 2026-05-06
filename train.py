import torch
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from data import prepare_data
from utils import MyDataset, compute_metrics, label2id
import os

def train_model(model_name='distilbert-base-cased', output_dir='./results'):
    # Enable W&B tracking for Task 4
    os.environ["WANDB_PROJECT"] = "mlops-assignment2"
    
    # Load and encode data
    train_encodings, train_labels, test_encodings, test_labels = prepare_data(model_name)
    
    # Create datasets
    train_dataset = MyDataset(train_encodings, train_labels)
    test_dataset = MyDataset(test_encodings, test_labels)
    
    # Load model
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=len(label2id))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Define training arguments with W&B reporting enabled
    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        output_dir=output_dir,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy='steps',
        report_to="wandb"  # Enabled W&B
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    trainer.train()
    trainer.save_model('distilbert-reviews-genres')
    return trainer

if __name__ == '__main__':
    train_model()
