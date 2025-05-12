import numpy as np
from scipy import stats
from typing import List, Dict

class DriftDetector:
    def __init__(self, reference_data: List[Dict]):
        self.reference_features = self._extract_features(reference_data)
        self.p_value_threshold = 0.05
        
    def _extract_features(self, data: List[Dict]) -> Dict:
        # Extract relevant features from the data
        features = {}
        for item in data:
            for key, value in item.items():
                if key not in features:
                    features[key] = []
                features[key].append(value)
        return features
    
    def detect_drift(self, current_data: List[Dict]) -> Dict:
        current_features = self._extract_features(current_data)
        drift_results = {}
        
        for feature in self.reference_features:
            if feature in current_features:
                # Perform Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(
                    self.reference_features[feature],
                    current_features[feature]
                )
                
                drift_results[feature] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < self.p_value_threshold
                }
                
        return drift_results