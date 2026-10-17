
import pandas as pd
import os
from sklearn.ensemble import IsolationForest

# Load the analyzed survey data
df = pd.read_csv("outputs/response_analysis.csv")

# Select numerical features for machine learning
features = [
    "Completion_Rate",
    "Unique_Answers",
    "Response_Variation",
    "Average_Rating",
    "Comment_Length",
    "Response_Time_Seconds"
]

# Prepare the data
X = df[features].copy()

# Handle missing and infinite values
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median())

# Create the Isolation Forest model
model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)

# Train the model and detect anomalies
predictions = model.fit_predict(X)

# Convert predictions into readable labels
df["Anomaly_Status"] = predictions
df["Anomaly_Status"] = df["Anomaly_Status"].map({
    1: "Normal",
    -1: "Anomaly"
})

# Display the results
print("\nAnomaly Detection Results:")
print(
    df[
        ["Response_ID", "Anomaly_Status"]
    ].head(10)
)

# Display summary
print("\nTotal Responses:", len(df))
print(
    "Normal Responses:",
    (df["Anomaly_Status"] == "Normal").sum()
)
print(
    "Anomalies Detected:",
    (df["Anomaly_Status"] == "Anomaly").sum()
)

# Save the results
os.makedirs("outputs", exist_ok=True)

df.to_csv("outputs/anomaly_results.csv", index=False)

print("\nAnomaly detection completed successfully!")
print("Results saved to outputs/anomaly_results.csv")