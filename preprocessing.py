
import pandas as pd
import os


def load_dataset(file_path):
    """Load survey responses from a CSV file."""

    try:
        df = pd.read_csv(file_path)

        print("\nDataset loaded successfully!")
        print("-" * 40)

        print("\nFirst 5 Responses:")
        print(df.head())

        print("\nDataset Shape:")
        print(f"Rows: {df.shape[0]}")
        print(f"Columns: {df.shape[1]}")

        return df

    except FileNotFoundError:
        print("Error: CSV file not found.")
        return None

    except Exception as error:
        print("Error loading dataset:", error)
        return None


def check_missing_values(df):
    """Check missing values in each column."""

    print("\nMissing Values:")
    print(df.isnull().sum())

    total_missing = df.isnull().sum().sum()
    print(f"\nTotal Missing Values: {total_missing}")


def check_duplicates(df):
    """Identify duplicate survey responses."""

    duplicate_count = df.duplicated().sum()

    print(f"\nDuplicate Rows: {duplicate_count}")


def clean_dataset(df):
    """Remove exact duplicate rows and prepare the dataset."""

    df = df.copy()

    initial_rows = len(df)

    df = df.drop_duplicates()

    removed_rows = initial_rows - len(df)

    print(f"\nDuplicate Rows Removed: {removed_rows}")
    print(f"Rows After Cleaning: {len(df)}")

    return df


def save_cleaned_dataset(df):
    """Save the cleaned dataset to the outputs folder."""

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/cleaned_survey_data.csv"

    df.to_csv(output_path, index=False)

    print(f"\nCleaned dataset saved to: {output_path}")


if __name__ == "__main__":

    file_path = "data/survey_response_quality_demo.csv"

    df = load_dataset(file_path)

    if df is not None:

        check_missing_values(df)

        check_duplicates(df)

        cleaned_df = clean_dataset(df)

        save_cleaned_dataset(cleaned_df)
