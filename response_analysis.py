import pandas as pd
import os

# Load the survey data
df = pd.read_csv("outputs/survey_features.csv")

# 1. Check incomplete responses
df["Incomplete_Response"] = df["Missing_Answers"] > 0

# 2. Check straight-lining
df["Straight_Lining_Flag"] = df["Straight_Lining"]

# 3. Check very fast responses
df["Fast_Response_Flag"] = df["Very_Short_Response_Time"]

# 4. Check repeated answer patterns
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

# Keep only question columns that exist
question_columns = [
    col for col in question_columns if col in df.columns
]

df["Repeated_Pattern_Flag"] = df.duplicated(
    subset=question_columns,
    keep=False
)

# 5. Check short comments
df["Short_Comment_Flag"] = (
    df["Comment_Length"] <= 2
)

# 6. Final review recommendation
df["Review_Recommended"] = (
    df["Incomplete_Response"]
    | df["Straight_Lining_Flag"]
    | df["Fast_Response_Flag"]
    | df["Repeated_Pattern_Flag"]
    | df["Short_Comment_Flag"]
)

# Display results
print("\nResponse Analysis Results:")
print(
    df[
        [
            "Response_ID",
            "Incomplete_Response",
            "Straight_Lining_Flag",
            "Fast_Response_Flag",
            "Repeated_Pattern_Flag",
            "Short_Comment_Flag",
            "Review_Recommended"
        ]
    ].head(10)
)

# Display summary
print("\nTotal Responses:", len(df))
print("Incomplete Responses:", df["Incomplete_Response"].sum())
print("Straight-Lining Responses:", df["Straight_Lining_Flag"].sum())
print("Fast Responses:", df["Fast_Response_Flag"].sum())
print("Repeated Patterns:", df["Repeated_Pattern_Flag"].sum())
print("Responses Recommended for Review:", df["Review_Recommended"].sum())

# Save the results
os.makedirs("outputs", exist_ok=True)

df.to_csv("outputs/response_analysis.csv", index=False)

print("\nAnalysis completed successfully!")
print("Results saved in outputs/response_analysis.csv")