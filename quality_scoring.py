import pandas as pd
import os

# Load anomaly detection results
df = pd.read_csv("outputs/anomaly_results.csv")

# Start with a score of 100 for every response
df["Quality_Score"] = 100

# Reduce score for incomplete responses
df.loc[df["Incomplete_Response"], "Quality_Score"] -= 20

# Reduce score for straight-lining
df.loc[df["Straight_Lining_Flag"], "Quality_Score"] -= 15

# Reduce score for very fast responses
df.loc[df["Fast_Response_Flag"], "Quality_Score"] -= 15

# Reduce score for repeated answer patterns
df.loc[df["Repeated_Pattern_Flag"], "Quality_Score"] -= 15

# Reduce score for very short comments
df.loc[df["Short_Comment_Flag"], "Quality_Score"] -= 10

# Reduce score if Isolation Forest detects an anomaly
df.loc[df["Anomaly_Status"] == "Anomaly", "Quality_Score"] -= 15

# Keep the score between 0 and 100
df["Quality_Score"] = df["Quality_Score"].clip(0, 100)

# Classify responses based on their scores
def classify_quality(score):
    if score >= 75:
        return "High Quality"
    elif score >= 50:
        return "Medium Quality"
    else:
        return "Low Quality"

df["Quality_Category"] = df["Quality_Score"].apply(classify_quality)

# Display results
print("\nSurvey Response Quality Results:\n")

print(
    df[
        ["Response_ID", "Quality_Score", "Quality_Category"]
    ].head(10).to_string(index=False)
)

# Display category summary
print("\nQuality Category Summary:")
print(df["Quality_Category"].value_counts())

# Save the final results
os.makedirs("outputs", exist_ok=True)

df.to_csv("outputs/final_quality_results.csv", index=False)

print("\nQuality scoring completed successfully!")
print("Results saved to outputs/final_quality_results.csv")
