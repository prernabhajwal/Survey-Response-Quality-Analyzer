import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df):

    # Create a copy of survey data
    df = df.copy()

    # Select features for anomaly detection
    features = [
        "Completion_Rate",
        "Unique_Answers",
        "Response_Variation",
        "Average_Rating",
        "Comment_Length",
        "Response_Time_Seconds"
    ]

    # Use available columns only
    features = [col for col in features if col in df.columns]

    # Select feature data
    X = df[features].copy()

    # Convert values into numbers
    for column in features:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    # Handle missing and infinite values
    X = X.replace([float("inf"), float("-inf")], 0)
    X = X.fillna(0)

    # Detect anomalies
    if len(df) >= 2 and len(features) > 0:

        model = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42
        )

        predictions = model.fit_predict(X)

        # Assign labels
        df["Anomaly_Status"] = [
            "Normal" if value == 1 else "Anomaly"
            for value in predictions
        ]

    else:
        df["Anomaly_Status"] = "Normal"

    return df


# Run this file independently
if __name__ == "__main__":

    # Load survey data
    df = pd.read_csv("outputs/response_analysis.csv")

    # Detect anomalies
    result = detect_anomalies(df)

    # Display results
    print("\nAnomaly Detection Results:")
    print(result[["Response_ID", "Anomaly_Status"]].head(10))

    print("\nTotal Responses:", len(result))
    print(
        "Normal Responses:",
        (result["Anomaly_Status"] == "Normal").sum()
    )
    print(
        "Anomalies Detected:",
        (result["Anomaly_Status"] == "Anomaly").sum()
    )

    # Save results
    result.to_csv("outputs/anomaly_results.csv", index=False)

    print("\nAnomaly detection completed successfully!")
    print("Results saved in outputs/anomaly_results.csv")

