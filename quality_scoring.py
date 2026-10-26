import pandas as pd


def score_quality(df):
    df = df.copy()
    df["Quality_Score"] = 100

    df.loc[df["Incomplete_Response"] == True, "Quality_Score"] -= 20
    df.loc[df["Straight_Lining_Flag"] == True, "Quality_Score"] -= 15
    df.loc[df["Fast_Response_Flag"] == True, "Quality_Score"] -= 15
    df.loc[df["Repeated_Pattern_Flag"] == True, "Quality_Score"] -= 15
    df.loc[df["Short_Comment_Flag"] == True, "Quality_Score"] -= 10
    df.loc[df["Anomaly_Status"] == "Anomaly", "Quality_Score"] -= 15

    df["Quality_Score"] = df["Quality_Score"].clip(0, 100)

    def classify_quality(score):
        if score >= 75:
            return "High Quality"
        elif score >= 50:
            return "Medium Quality"
        else:
            return "Low Quality"

    df["Quality_Category"] = df["Quality_Score"].apply(classify_quality)

    return df


if __name__ == "__main__":
    df = pd.read_csv("outputs/anomaly_results.csv")
    result = score_quality(df)

    print(result[["Response_ID", "Quality_Score", "Quality_Category"]].to_string(index=False))
    print("\nQuality Category Summary:")
    print(result["Quality_Category"].value_counts())

    result.to_csv("outputs/final_quality_results.csv", index=False)
    print("\nQuality scoring completed successfully!")
