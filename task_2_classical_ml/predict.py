import pandas as pd
import joblib

def main():
    predict_df = pd.read_csv("./data/hidden_test.csv")   
    loaded_model = joblib.load('./models/xgb_model.pkl')

    y_pred = loaded_model.predict(predict_df)  
    
    prediction_df = pd.DataFrame(y_pred, columns=["target"])
    prediction_df.to_csv("./data/predictions.csv")
    print("Predictions saved to ./data/predictions.csv")
    
if __name__ == "__main__":
    main()