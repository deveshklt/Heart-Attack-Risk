import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse

def preprocess_data(input_file_path, output_df_path, output_csv_path):
    # Load the dataset
    df = pd.read_csv(input_file_path)

    # Ensure the "Blood Pressure" column is treated as a string
    df['Blood Pressure'] = df['Blood Pressure'].astype(str)

    # Split the blood pressure data into systolic and diastolic columns
    bp_split = df['Blood Pressure'].str.split('/', expand=True)

    # Create a new column for systolic blood pressure
    df['Systolic_BP'] = bp_split[0].astype(int)

    # Create a new column for diastolic blood pressure if available
    if bp_split.shape[1] > 1:
        df['Diastolic_BP'] = bp_split[1].astype(int)

    df.drop(['Blood Pressure'], axis=1, inplace=True)

    # Calculate Mean Arterial Pressure (MAP)
    df['Blood Pressure'] = (2/3 * df['Diastolic_BP']) + (1/3 * df['Systolic_BP'])

    # Drop the original blood pressure, systolic, and diastolic columns
    df.drop(['Systolic_BP', 'Diastolic_BP'], axis=1, inplace=True)

    # Create a LabelEncoder instance
    label_encoder = LabelEncoder()

    # Fit and transform the columns you want to encode
    df['Patient ID'] = label_encoder.fit_transform(df['Patient ID'])
    df['Diet'] = label_encoder.fit_transform(df['Diet'])
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    df['Continent'] = label_encoder.fit_transform(df['Continent'])
    df['Hemisphere'] = label_encoder.fit_transform(df['Hemisphere'])
    df['Country'] = label_encoder.fit_transform(df['Country'])
    df['Blood Pressure'] = label_encoder.fit_transform(df['Blood Pressure'])

    # Save the preprocessed DataFrame
    df.to_pickle(output_df_path)

    # Save the preprocessed data to a CSV file
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess raw data and save the preprocessed data.")
    parser.add_argument("--input_file_path", required=True, help="Path to the raw input CSV file.")
    parser.add_argument("--output_df_path", required=True, help="Path to save the preprocessed DataFrame.")
    parser.add_argument("--output_csv_path", required=True, help="Path to save the preprocessed data CSV file.")
    args = parser.parse_args()

    # Replace placeholder paths with actual paths
    input_file_path = args.input_file_path
    output_df_path = args.output_df_path
    output_csv_path = args.output_csv_path

    # Example usage:
    # python train_data_preprocessing.py --input_file_path path/to/your/input/data.csv --output_df_path path/to/your/output/preprocessed_data.pkl --output_csv_path path/to/your/output/preprocessed_data.csv

    # Print paths to verify correctness
    print("Input File Path:", input_file_path)
    print("Output DataFrame Path:", output_df_path)
    print("Output CSV Path:", output_csv_path)

    # Preprocess the data and save both the DataFrame and the CSV file
    preprocess_data(input_file_path, output_df_path, output_csv_path)
