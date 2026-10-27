import pandas as pd


# Function to analyze survey responses
def analyze_responses(df, question_columns=None):

    # Create a copy of survey data
    df = df.copy()
   
    print("\n--- CHECK BEFORE ANALYSIS ---")
    print(df["Missing_Answers"].value_counts(dropna=False))
    print("Total missing cells:", df["Missing_Answers"].sum())


    # 1. Check incomplete responses
    df["Incomplete_Response"] = df["Missing_Answers"] > 0

    print("\n--- INCOMPLETE RESPONSE CHECK ---")
    print(df["Missing_Answers"].value_counts().sort_index())
    print("Incomplete responses:", df["Incomplete_Response"].sum())
    print("Complete responses:", (df["Missing_Answers"] == 0).sum())

    # 2. Check straight-lining
    df["Straight_Lining_Flag"] = df["Straight_Lining"]

    # 3. Check very fast responses
    df["Fast_Response_Flag"] = df["Very_Short_Response_Time"]

    # 4. Check repeated answer patterns
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

    if question_columns:
        df["Repeated_Pattern_Flag"] = df.duplicated(
            subset=question_columns,
            keep=False
        )
    else:
        df["Repeated_Pattern_Flag"] = False

    # 5. Check short comments
    if "Comment_Length" in df.columns:
        df["Short_Comment_Flag"] = df["Comment_Length"] <= 2
    else:
        df["Short_Comment_Flag"] = False

    # 6. Check potentially contradictory responses
    df["Contradictory_Response_Flag"] = False

    required_columns = [
        "Q1_Ease_of_Use",
        "Q5_Clarity",
        "Q2_Satisfaction",
        "Q8_Overall_Experience",
        "Q4_Support"
    ]

    if all(col in df.columns for col in required_columns):

        df["Contradictory_Response_Flag"] = (
            (
                (df["Q1_Ease_of_Use"] <= 2)
                & (df["Q5_Clarity"] >= 4)
            )
            | (
                (df["Q1_Ease_of_Use"] >= 4)
                & (df["Q5_Clarity"] <= 2)
            )
            | (
                (df["Q2_Satisfaction"] <= 2)
                & (df["Q8_Overall_Experience"] >= 4)
            )
            | (
                (df["Q2_Satisfaction"] >= 4)
                & (df["Q8_Overall_Experience"] <= 2)
            )
            | (
                (df["Q4_Support"] <= 2)
                & (df["Q8_Overall_Experience"] >= 4)
            )
            | (
                (df["Q4_Support"] >= 4)
                & (df["Q8_Overall_Experience"] <= 2)
            )
        )

    # 7. Recommend responses for review
    df["Review_Recommended"] = (
        df["Incomplete_Response"]
        | df["Straight_Lining_Flag"]
        | df["Fast_Response_Flag"]
        | df["Repeated_Pattern_Flag"]
        | df["Short_Comment_Flag"]
        | df["Contradictory_Response_Flag"]
    )
    print("\n--- Flag Summary ---")
    flags = [
        "Incomplete_Response",
        "Straight_Lining_Flag",
        "Fast_Response_Flag",
        "Repeated_Pattern_Flag",
        "Short_Comment_Flag",
        "Contradictory_Response_Flag"
    ]

    for col in flags:
        print(f"{col}: {df[col].sum()}")

    print("Total Responses:", len(df))
    print("Review Recommended:", df["Review_Recommended"].sum())
    return df


# Run this file independently
if __name__ == "__main__":

    # Load survey features
    df = pd.read_csv("outputs/survey_features.csv")

    # Analyze responses
    result = analyze_responses(df)

    # Display only important columns
    result_columns = [
        "Response_ID",
        "Incomplete_Response",
        "Straight_Lining_Flag",
        "Fast_Response_Flag",
        "Repeated_Pattern_Flag",
        "Short_Comment_Flag",
        "Contradictory_Response_Flag",
        "Review_Recommended"
    ]

    result_columns = [
        col for col in result_columns if col in result.columns
    ]

    print("\nResponse Analysis Results:")
    print(result[result_columns].head(10).to_string(index=False))

    # Display summary
    print("\nTotal Responses:", len(result))
    print("Incomplete Responses:", result["Incomplete_Response"].sum())
    print("Straight-Lining Responses:", result["Straight_Lining_Flag"].sum())
    print("Fast Responses:", result["Fast_Response_Flag"].sum())
    print("Repeated Patterns:", result["Repeated_Pattern_Flag"].sum())
    print("Short Comments:", result["Short_Comment_Flag"].sum())
    print(
        "Potentially Contradictory Responses:",
        result["Contradictory_Response_Flag"].sum()
    )
    print("Responses Recommended for Review:", result["Review_Recommended"].sum())

    # Save results
    result.to_csv("outputs/response_analysis.csv", index=False)

    print("\nAnalysis completed successfully!")
    print("Results saved in outputs/response_analysis.csv")

