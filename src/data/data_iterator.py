from torch.utils.data import DataLoader
from .data_generator import TextDataGenerator

class DataIterator:
    def __init__(self, tokenizer, batch_size=32, max_length=128):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        
    def get_train_iterator(self, train_file):
        train_dataset = TextDataGenerator(
            train_file,
            self.tokenizer,
            self.max_length
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
    def get_test_iterator(self, test_file):
        test_dataset = TextDataGenerator(
            test_file,
            self.tokenizer,
            self.max_length
        )
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )