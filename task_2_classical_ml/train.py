from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib
import json


def main():
    print("Reading the data")
    train_df = pd.read_csv("./data/train.csv")   
    X, y = train_df.drop(columns=["target"]), train_df[["target"]] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   

    with open("./params/param_grid.json", "r") as f:
        param_grid = json.load(f)

    print("Using following param grid")
    print(param_grid) 
    
    regressor = XGBRegressor(random_state=42)

    print("Performing Cross Validation to check the best parameters")
    
    grid_search = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5, 
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV RMSE:", np.sqrt(-grid_search.best_score_))
    
    best_params = grid_search.best_params_

    best_xgb = XGBRegressor(**best_params, random_state=42)

    best_xgb.fit(X_train, y_train)

    y_pred = best_xgb.predict(X_test)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {test_rmse:.4f}")

    print("Saving the model")
    
    joblib.dump(best_xgb, './models/xgb_model.pkl')
    print("Model saved as './models/xgb_model.pkl'")
    
if __name__ == "__main__":
    main()