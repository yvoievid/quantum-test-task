import numpy as np
import json

class SimpleLinearRegression:
    def __init__(self, coefficients=None, intercept=0):
        self.coefficients = coefficients 
        self.intercept = intercept  

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X] 
        self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept = self.coefficients[0]  
        self.coefficients = self.coefficients[1:] 

    def predict(self, X):
        return X.dot(self.coefficients) + self.intercept

    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = np.mean((y - predictions) ** 2)
        return mse
    
    def save_model(self, file_path):
        model_params = {
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist()
        }
        with open(file_path, 'w') as file:
            json.dump(model_params, file)

