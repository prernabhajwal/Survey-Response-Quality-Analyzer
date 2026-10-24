import pandas as pd


def extract_features(df):

    # Protect original dataset
    df = df.copy()

    # Select survey questions
    questions = df.loc[:, "Q1_Ease_of_Use":"Q8_Overall_Experience"]

    # Convert answers into numbers
    questions = questions.apply(pd.to_numeric, errors="coerce")

    # Count missing answers
    df["Missing_Answers"] = questions.isnull().sum(axis=1)

    # Calculate completion rate
    df["Completion_Rate"] = (
        (8 - df["Missing_Answers"]) / 8
    ) * 100

    # Count different answers
    df["Unique_Answers"] = questions.nunique(axis=1)

    # Calculate average rating
    df["Average_Rating"] = questions.mean(axis=1)

    # Calculate rating variation
    df["Response_Variation"] = questions.std(axis=1).fillna(0)

    # Detect same answers
    answered = questions.notnull().sum(axis=1)

    df["Straight_Lining"] = (
        (df["Unique_Answers"] == 1) & (answered >= 2)
    )
    # Check response time
    if "Response_Time_Seconds" in df.columns:
        df["Very_Short_Response_Time"] = (
            pd.to_numeric(df["Response_Time_Seconds"], errors="coerce") < 60
        )
    else:
        df["Very_Short_Response_Time"] = False
        
    # Count words in comments
    df["Comment_Length"] = (
        df["Free_Text_Comment"].fillna("").str.split().str.len()
    )

    return df


if __name__ == "__main__":

    df = pd.read_csv("outputs/cleaned_survey_data.csv")

    result = extract_features(df)

    print("\nFeature Engineering Results:")
    print(result.head())

    result.to_csv("outputs/survey_features.csv", index=False)

    print("\nFeatures saved successfully!")
