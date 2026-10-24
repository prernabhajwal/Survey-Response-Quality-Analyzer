import pandas as pd

# =====================================
# 1. EXPLORATORY DATA ANALYSIS (EDA)
# =====================================

def perform_eda(df):

    print("\n========== DATASET OVERVIEW ==========")
    print("Dataset Shape:", df.shape)

    print("\nFirst Five Responses:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nStatistical Summary:")
    print(df.describe(include="all"))

    print("\nMean:") 
    print(df.select_dtypes(include="number").mean())
    print("\nMedian:") 
    print(df.select_dtypes(include="number").median())
    print("\nMode:") 
    print(df.mode().iloc[0])

# =====================================
# 2. DATA PREPROCESSING
# =====================================

def preprocess_data(df):

    if df is None or df.empty:
        raise ValueError("Please upload a valid survey dataset.")

    # Create a copy
    df = df.copy()

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    # Remove completely empty rows and columns
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # Remove duplicate responses
    df.drop_duplicates(inplace=True)

    # Remove extra spaces from text responses
    for column in df.select_dtypes(include=["object", "str"]).columns:
        df[column] = df[column].apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    # Convert blank strings into missing values
    df.replace(r"^\s*$", pd.NA, regex=True, inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    return df


# =====================================
# 3. RUN EDA AND PREPROCESSING
# =====================================

if __name__ == "__main__":

    # Read the original survey dataset
    df = pd.read_csv("data/survey_response_quality_demo.csv")

    # Step 1: Perform EDA
    perform_eda(df)

    # Step 2: Clean the dataset
    cleaned_df = preprocess_data(df)

    # Step 3: Save cleaned data
    cleaned_df.to_csv("outputs/cleaned_survey_data.csv", index=False)

    print("\nPreprocessing completed successfully!")
    print("Cleaned dataset saved as cleaned_survey_data.csv")

    print("\nCleaned Dataset:")
    print(cleaned_df.head())

