import pandas as pd


def extract_features(df, question_columns=None):

    # Protect original dataset
    df = df.copy()

    # Select survey questions
    if question_columns is None:
        question_columns = [
            "Q1_Ease_of_Use",
            "Q2_Satisfaction",
            "Q3_Speed",
            "Q4_Support",
            "Q5_Clarity",
            "Q6_Reliability",
            "Q7_Recommend",
            "Q8_Overall_Experience"
        ]

    question_columns = [
        col for col in question_columns if col in df.columns
    ]

    if not question_columns:
        raise ValueError("Please select survey question columns.")

    questions = df[question_columns].copy()

    # Convert answers into numbers
    questions = questions.apply(pd.to_numeric, errors="coerce")

    # Count missing answers
    df["Missing_Answers"] = questions.isnull().sum(axis=1)

    # Calculate completion rate
    df["Completion_Rate"] = (
        (len(question_columns) - df["Missing_Answers"])
        / len(question_columns)
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
            pd.to_numeric(
                df["Response_Time_Seconds"], errors="coerce"
            ) < 60
        )
    else:
        df["Very_Short_Response_Time"] = False

    # Count words in comments
    if "Free_Text_Comment" in df.columns:
        df["Comment_Length"] = (
            df["Free_Text_Comment"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )
    else:
        df["Comment_Length"] = 0

    return df


if __name__ == "__main__":

    df = pd.read_csv("outputs/cleaned_survey_data.csv")

    result = extract_features(df)

    print("\nFeature Engineering Results:")
    print(result.head())

    result.to_csv("outputs/survey_features.csv", index=False)

    print("\nFeatures saved successfully!")
