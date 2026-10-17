
import pandas as pd
import os


def extract_features(df):
    """Extract useful features from survey responses."""

    df = df.copy()

    # Survey question columns
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

    # Keep only question columns that exist in the dataset
    question_columns = [
        col for col in question_columns if col in df.columns
    ]

    if not question_columns:
        raise ValueError("No survey question columns found.")

    # Convert answers to numeric values
    answers = df[question_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # 1. Completion Rate (%)
    total_questions = len(question_columns)

    df["Completion_Rate"] = (
        answers.notna().sum(axis=1) / total_questions
    ) * 100

    # 2. Number of Unique Answers
    df["Unique_Answers"] = answers.nunique(axis=1)

    # 3. Response Variation
    df["Response_Variation"] = answers.std(axis=1).fillna(0)

    # 4. Straight-Lining Detection
    # True when all answered questions have the same value
    # and at least two questions have been answered.
    answered_count = answers.notna().sum(axis=1)

    df["Straight_Lining"] = (
        (df["Unique_Answers"] == 1) &
        (answered_count >= 2)
    )

    # 5. Average Survey Rating
    df["Average_Rating"] = answers.mean(axis=1)

    # 6. Free-Text Comment Length
    if "Free_Text_Comment" in df.columns:
        comments = df["Free_Text_Comment"].fillna("").astype(str)

        df["Comment_Length"] = comments.apply(
            lambda text: len(text.split())
        )
    else:
        df["Comment_Length"] = 0

    # 7. Missing Answers
    df["Missing_Answers"] = answers.isna().sum(axis=1)

    # 8. Very Short Response Time
    if "Response_Time_Seconds" in df.columns:
        response_time = pd.to_numeric(
            df["Response_Time_Seconds"], errors="coerce"
        )

        df["Very_Short_Response_Time"] = (
            response_time < 60
        ).fillna(False)
    else:
        df["Very_Short_Response_Time"] = False

    return df


def save_features(df):
    """Save extracted features to the outputs folder."""

    os.makedirs("outputs", exist_ok=True)

    output_path = "outputs/survey_features.csv"

    df.to_csv(output_path, index=False)

    print(f"\nFeature data saved to: {output_path}")


if __name__ == "__main__":

    input_path = "outputs/cleaned_survey_data.csv"

    try:
        df = pd.read_csv(input_path)

        print("Dataset loaded successfully!")

        feature_df = extract_features(df)

        print("\nExtracted Features:")
        print(
            feature_df[
                [
                    "Response_ID",
                    "Completion_Rate",
                    "Unique_Answers",
                    "Response_Variation",
                    "Straight_Lining",
                    "Average_Rating",
                    "Comment_Length",
                    "Missing_Answers",
                    "Very_Short_Response_Time"
                ]
            ].head(10).to_string(index=False)
        )

        save_features(feature_df)

        print("\nFeature engineering completed successfully!")

    except FileNotFoundError:
        print("Error: Cleaned dataset not found.")
        print("Run preprocessing.py first.")

    except Exception as error:
        print("Error:", error)
