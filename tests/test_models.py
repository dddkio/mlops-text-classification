import pytest
import torch
import numpy as np
from transformers import BertTokenizer
from src.models.bert_model import BertClassifier
from src.models.factory import ModelFactory

@pytest.fixture
def bert_model():
    model = BertClassifier(num_classes=13)
    model.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model

def test_model_factory():
    try:
        model = ModelFactory.get_model('bert', num_classes=13)
        assert isinstance(model, BertClassifier)
    except Exception as e:
        pytest.fail(f"Model factory failed to create model: {str(e)}")
    
    with pytest.raises(ValueError):
        ModelFactory.get_model('invalid_model')

def test_bert_model_structure(bert_model):
    # Test model attributes
    assert hasattr(bert_model, 'model')
    assert hasattr(bert_model, 'device')
    
    # Test required methods
    assert hasattr(bert_model, 'train')
    assert hasattr(bert_model, 'predict')
    assert hasattr(bert_model, 'evaluate')
    assert hasattr(bert_model, 'save')
    assert hasattr(bert_model, 'load')

def test_bert_model_predict(bert_model):
    test_text = "This is a test sentence"
    prediction = bert_model.predict(test_text)
    assert isinstance(prediction, dict)
    assert 'predictions' in prediction
    assert isinstance(prediction['predictions'], np.ndarray)

def test_bert_model_training(bert_model):
    # Mock training data
    mock_data = [
        {
            'input_ids': torch.randint(0, 1000, (2, 128)),
            'attention_mask': torch.ones(2, 128),
            'label': torch.tensor([0, 1])
        }
    ]
    
    # Test training
    bert_model.train(mock_data)
    # Since train() doesn't return anything, we just verify it runs without errors