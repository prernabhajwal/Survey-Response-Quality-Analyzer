import pandas as pd
from sklearn.ensemble import IsolationForest

# Step 1: Load survey data
df = pd.read_csv("outputs/response_analysis.csv")


# Step 2: Select features for anomaly detection
features = [
    "Completion_Rate",
    "Unique_Answers",
    "Response_Variation",
    "Average_Rating",
    "Comment_Length",
    "Response_Time_Seconds"
]

X = df[features].copy()


# Step 3: Convert feature values into numbers
X = X.apply(pd.to_numeric, errors="coerce")


# Step 4: Handle missing and infinite values
X = X.replace([float("inf"), float("-inf")], float("nan"))

X = X.fillna(X.median())
X = X.fillna(0)


# Step 5: Create the Isolation Forest model
model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42
)


# Step 6: Detect anomalies
predictions = model.fit_predict(X)


# Step 7: Assign labels to responses
df["Anomaly_Status"] = predictions

df["Anomaly_Status"] = df["Anomaly_Status"].replace({
    1: "Normal",
    -1: "Anomaly"
})


# Step 8: Display the first 10 results
print("\nAnomaly Detection Results:")

print(
    df[["Response_ID", "Anomaly_Status"]].head(10)
)


# Step 9: Display only anomalous responses
print("\nDetails of Anomalous Responses:")

anomalies = df[df["Anomaly_Status"] == "Anomaly"]

print(
    anomalies[
        ["Response_ID"] + features
    ].to_string(index=False)
)


# Step 10: Display summary
print("\nTotal Responses:", len(df))

print(
    "Normal Responses:",
    (df["Anomaly_Status"] == "Normal").sum()
)

print(
    "Anomalies Detected:",
    (df["Anomaly_Status"] == "Anomaly").sum()
)


# Step 11: Save the results
df.to_csv("outputs/anomaly_results.csv", index=False)

print("\nAnomaly detection completed successfully!")
print("Results saved to outputs/anomaly_results.csv")

