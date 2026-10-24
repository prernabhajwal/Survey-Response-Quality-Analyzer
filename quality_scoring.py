import pandas as pd

# Step 1: Load anomaly detection results
df = pd.read_csv("outputs/anomaly_results.csv")


# Step 2: Give every response a starting score of 100
df["Quality_Score"] = 100


# Step 3: Deduct points for response quality issues

# Incomplete responses
df.loc[df["Incomplete_Response"] == True, "Quality_Score"] -= 20

# Straight-lining
df.loc[df["Straight_Lining_Flag"] == True, "Quality_Score"] -= 15

# Very fast responses
df.loc[df["Fast_Response_Flag"] == True, "Quality_Score"] -= 15

# Repeated answer patterns
df.loc[df["Repeated_Pattern_Flag"] == True, "Quality_Score"] -= 15

# Very short comments
df.loc[df["Short_Comment_Flag"] == True, "Quality_Score"] -= 10

# Anomalies detected by Isolation Forest
df.loc[df["Anomaly_Status"] == "Anomaly", "Quality_Score"] -= 15


# Step 4: Keep scores between 0 and 100
df["Quality_Score"] = df["Quality_Score"].clip(0, 100)


# Step 5: Classify response quality
def classify_quality(score):
    if score >= 75:
        return "High Quality"
    elif score >= 50:
        return "Medium Quality"
    else:
        return "Low Quality"


df["Quality_Category"] = df["Quality_Score"].apply(classify_quality)

# Step 6: Display all response quality results
print("\nAll Survey Response Quality Results:\n")

print(
    df[
        ["Response_ID", "Quality_Score", "Quality_Category"]
    ].to_string(index=False)
)


# Step 7: Display category summary
print("\nQuality Category Summary:")
print(df["Quality_Category"].value_counts())


# Step 8: Save final results
df.to_csv("outputs/final_quality_results.csv", index=False)

print("\nQuality scoring completed successfully!")
