from abc import ABC, abstractmethod

class DigitClassificationInterface(ABC):
    @abstractmethod
    def fit(self, features, labels):
        pass

    @abstractmethod
    def predict(self, input):
        pass
