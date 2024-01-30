import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import argparse

def train_model(X_train, X_test, y_train, y_test, model_output_directory):
    # Choose a machine learning algorithm
    model = RandomForestClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model in the model directory
    model_output_path = os.path.join(model_output_directory, "random_forest_model.pkl")
    joblib.dump(model, model_output_path)

    # Make predictions on the training set
    y_train_pred = model.predict(X_train)

    # Evaluate the model on the training set
    accuracy_train = accuracy_score(y_train, y_train_pred)
    classification_rep_train = classification_report(y_train, y_train_pred)

    # Print the evaluation metrics for the training set
    print("Training Set Metrics:")
    print(f"Accuracy: {accuracy_train}")
    print("Classification Report:\n", classification_rep_train)

    # Make predictions on the testing set
    y_test_pred = model.predict(X_test)

    # Evaluate the model on the testing set
    accuracy_test = accuracy_score(y_test, y_test_pred)
    classification_rep_test = classification_report(y_test, y_test_pred)

    # Print the evaluation metrics for the testing set
    print("\nTesting Set Metrics:")
    print(f"Accuracy: {accuracy_test}")
    print("Classification Report:\n", classification_rep_test)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument("--preprocessed_data_path", required=True, help="Path to the preprocessed data CSV file.")
    parser.add_argument("--model_output_directory", required=True, help="Directory to save the trained model.")
    args = parser.parse_args()

    # Example usage:
    # python train.py --preprocessed_data_path path/to/your/output/preprocessed_data.pkl --model_output_directory path/to/your/output/model

    # Load preprocessed data
    preprocessed_df = pd.read_pickle(args.preprocessed_data_path)

    # Define features (X) and labels (y) for training
    X = preprocessed_df.drop('Heart Attack Risk', axis=1)
    y = preprocessed_df['Heart Attack Risk']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model and print training and testing set metrics
    train_model(X_train, X_test, y_train, y_test, args.model_output_directory)
