from transformers import BertTokenizer
import torch
from src.data.data_generator import TextDataGenerator
def test_data_generator_content():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    generator = TextDataGenerator('data/train/train.csv', tokenizer)
    # Test actual data from your dataset
    item = generator[0]  # Get first itemSSSS
    
    # Test if it matches your actual data structure
    assert item['label'] in range(13)  # Should match your actual label range
     
    # Get original tweet and compare processed version
    original_tweet = generator.data.iloc[0]['tweet']
    processed_encoding = generator.tokenizer.encode_plus(
        original_tweet,
        add_special_tokens=True,
        max_length=generator.max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Verify the processing matches
    assert torch.equal(item['input_ids'], processed_encoding['input_ids'].flatten())
    assert torch.equal(item['attention_mask'], processed_encoding['attention_mask'].flatten())