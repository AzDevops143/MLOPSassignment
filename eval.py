import torch
import os
from transformers import DistilBertForSequenceClassification, Trainer
from data import prepare_data
from utils import MyDataset, compute_metrics, label2id
from sklearn.metrics import classification_report

def evaluate_model(model_path='distilbert-reviews-genres'):
    # Ensure we look in the absolute path if relative fails
    full_path = os.path.abspath(model_path)
    if not os.path.exists(full_path):
        print(f"Error: Model directory {full_path} not found. Please run train.py first.")
        return

    # Load the test data
    _, _, test_encodings, test_labels = prepare_data()
    test_dataset = MyDataset(test_encodings, test_labels)

    # Load the trained model
    model = DistilBertForSequenceClassification.from_pretrained(full_path)

    # Initialize trainer for evaluation
    trainer = Trainer(model=model, compute_metrics=compute_metrics)

    # Get predictions
    results = trainer.evaluate(test_dataset)
    print("Evaluation Results:", results)

    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)

    # Map back to labels
    id2label = {v: k for k, v in label2id.items()}
    true_names = [id2label[l] for l in test_labels]
    pred_names = [id2label[p] for p in preds]

    print("\nClassification Report:\n")
    print(classification_report(true_names, pred_names))

if __name__ == '__main__':
    evaluate_model()
