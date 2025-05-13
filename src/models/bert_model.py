import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from .base_model import BaseModel
class BertClassifier(BaseModel):
    def __init__(self, num_classes=13):  # 13 classes based on your data
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_classes
        ).to(self.device)
        
    def train(self, train_data, validation_data=None):
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        for epoch in range(3):  # Number of epochs
            for batch in train_data:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
    def predict(self, text: str) -> dict:
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'].to(self.device),
                attention_mask=encoding['attention_mask'].to(self.device)
            )
            
        predictions = torch.softmax(outputs.logits, dim=1)
        return {'prediction': predictions.cpu().numpy().tolist()[0]}
    
    def evaluate(self, test_data):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_data:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(test_data),
            'accuracy': correct / total
        }
    
    def save(self, path: str):
        self.model.save_pretrained(path)
        
    def load(self, path: str):
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)