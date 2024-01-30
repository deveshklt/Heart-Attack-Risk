import pandas as pd
import joblib
import argparse

def predict_and_save(input_test_data_path, model_path, output_prediction_csv_path):
    # Load preprocessed test data
    preprocessed_test_df = pd.read_csv(input_test_data_path)
    X_test = preprocessed_test_df

    # Load the trained model
    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Create a DataFrame for predictions
    prediction_df = pd.DataFrame({
        'Patient ID': preprocessed_test_df['Patient ID'],
        'Heart Attack Risk Prediction': y_pred
    })

    # Save predictions to CSV
    prediction_df.to_csv(output_prediction_csv_path, index=False)

    print("Predictions saved successfully.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict Heart Attack risk using a trained model.")
    parser.add_argument("--input_test_data_path", required=True, help="Path to the preprocessed test data CSV file.")
    parser.add_argument("--model_path", required=True, help="Path to the trained model file.")
    parser.add_argument("--output_prediction_csv_path", required=True, help="Path to save the predictions CSV file.")
    args = parser.parse_args()

    # Example usage:
    # python predict.py --input_test_data_path path/to/your/output/preprocessed_test_data.csv --model_path path/to/your/output/model/random_forest_model.pkl --output_prediction_csv_path path/to/your/output/predictions.csv

    # Predict and save the results
    predict_and_save(args.input_test_data_path, args.model_path, args.output_prediction_csv_path)
