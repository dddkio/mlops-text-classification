import pandas as pd
from typing import List, Dict
import re
import html

class TextPreprocessor:
    def __init__(self):
        # Initialize mappings for labels and intensities
        self.class_intensity_map = {
            'anger_low': 1, 'anger_medium': 2, 'anger_high': 3,
            'fear_low': 4, 'fear_medium': 5, 'fear_high': 6,
            'joy_low': 7, 'joy_medium': 8, 'joy_high': 9,
            'sadness_low': 10, 'sadness_medium': 11, 'sadness_high': 12
        }
        
    def clean_text(self, text: str) -> str:
        # Handle HTML entities (like &amp;)
        text = html.unescape(text)
        
        # Remove user mentions (@username)
        text = re.sub(r'@[\w]+', '', text)
        
        # Remove URLs if present
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Handle hashtags (remove # but keep the text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def preprocess_data(self, texts: List[str]) -> List[str]:
        """Process a list of texts"""
        return [self.clean_text(text) for text in texts]
    
    def get_label_from_intensity(self, class_intensity: str) -> int:
        """Convert class_intensity to numeric label"""
        return self.class_intensity_map.get(class_intensity, 0)
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire dataframe with all necessary transformations"""
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Clean tweets
        processed_df['processed_tweet'] = processed_df['tweet'].apply(self.clean_text)
        
        # Convert class_intensity to numeric labels if needed
        if 'class_intensity' in processed_df.columns:
            processed_df['numeric_label'] = processed_df['class_intensity'].apply(self.get_label_from_intensity)
            
        return processed_df