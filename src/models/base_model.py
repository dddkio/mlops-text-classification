from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModel(ABC):
    @abstractmethod
    def train(self, train_data, validation_data=None):
        pass

    @abstractmethod
    def predict(self, text: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def evaluate(self, test_data):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass