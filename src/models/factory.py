from .bert_model import BertClassifier

class ModelFactory:
    @staticmethod
    def get_model(model_type: str, **kwargs):
        models = {
            'bert': BertClassifier
        }
        
        if model_type not in models:
            raise ValueError(f"Model type {model_type} not supported")
            
        return models[model_type](**kwargs)