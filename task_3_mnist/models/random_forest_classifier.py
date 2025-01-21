
from interfaces.idigit_classifier import DigitClassificationInterface
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RFClassifier(DigitClassificationInterface):
    def __init__(self, random_state=42, n_estimators=100, max_depth=5):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=max_depth)

    def predict(self, image: np.ndarray):
        assert image.size == 784, f"Expected flattened array of size 784, got {image.size}"
        predictions = self.model.predict(image.reshape(1, -1))
        return predictions[0]
    
    def fit(self, features, labels):
        self.model.fit(features, labels)
        
