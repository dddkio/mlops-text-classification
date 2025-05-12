from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MetricsTracker:
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
    
    def update_training_metrics(self, train_loss: float, val_loss: float = None):
        self.metrics_history['train_loss'].append(train_loss)
        if val_loss:
            self.metrics_history['val_loss'].append(val_loss)
    
    def update_evaluation_metrics(self, y_true: List, y_pred: List):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        self.metrics_history['accuracy'].append(accuracy)
        self.metrics_history['precision'].append(precision)
        self.metrics_history['recall'].append(recall)
        self.metrics_history['f1'].append(f1)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def get_latest_metrics(self) -> Dict:
        return {
            metric: values[-1] if values else None
            for metric, values in self.metrics_history.items()
        }