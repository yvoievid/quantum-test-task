from models.cnn import CNN
from models.random_classifier import RandomValueClassifier
from models.random_forest_classifier import RFClassifier

class DigitClassifier:
    def __init__(self, algorithm="cnn"):
        if algorithm == "cnn":
            self.classifier = CNN()
        elif algorithm == "rf":
            self.classifier = RFClassifier()
        elif algorithm == "rand":
            self.classifier = RandomValueClassifier()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def predict(self, input):
        return self.classifier.predict(input)
    
    def fit(self, features, labels):
        print("Training classifier, wait please")
        self.classifier.fit(features, labels)
        print("Classifier is ready")
