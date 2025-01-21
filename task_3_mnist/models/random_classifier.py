
from interfaces.idigit_classifier import DigitClassificationInterface
import numpy as np

class RandomValueClassifier(DigitClassificationInterface):
    def __init__(self):
        super().__init__()

    def predict(self, image: np.ndarray):
        assert image.shape == (28, 28, 1), f"Expected shape (28, 28, 1), got {image.shape}"
        cropped_image = image[9:19, 9:19, 0]  # Center crop
        # Random bullshit go!
        return np.random.randint(0, 10)  # Return a random digit

    def fit(self, features, labels):
        raise NotImplementedError("The `fit` method must be implemented by the subclass.")