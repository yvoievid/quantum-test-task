import pandas as pd
import json
from models.linear_regression import SimpleLinearRegression

def main():
    with open("./params/linear_model.json", "r") as file:
        model_params = json.load(file)
    
    model = SimpleLinearRegression(
        intercept=model_params["intercept"],
        coefficients=model_params["coefficients"]
    )

    test_df = pd.read_csv("./data/hidden_test.csv") 
    X_new = test_df[['6', '7']].values      # Select important features
    predictions = model.predict(X_new)

    test_df['predictions'] = predictions
    test_df.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv")
    
if __name__ == "__main__":
    main()