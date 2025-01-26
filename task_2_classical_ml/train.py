from sklearn.model_selection import train_test_split
import pandas as pd
from models.linear_regression import SimpleLinearRegression

def main():
    print("Reading the data")
    train_df = pd.read_csv("./data/train.csv")   
    X = train_df[['6', '7']].values 
    y = train_df['target'].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SimpleLinearRegression()
    model.fit(X_train, y_train)

    print(f"Intercept (b): {model.intercept}")
    print(f"Coefficients (slopes): {model.coefficients}")

    model.save_model("./params/linear_model.json")
    print("Model saved to linear_model.json")

    mse = model.evaluate(X_test, y_test)
    print(f"Mean Squared Error on Test Data: {mse}")

if __name__ == "__main__":
    main()