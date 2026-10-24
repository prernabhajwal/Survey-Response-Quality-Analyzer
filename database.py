import sqlite3
import pandas as pd

DATABASE = "survey_database.db"


# 1. Create Table
def create_table():
    conn = sqlite3.connect(DATABASE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS survey_responses (
            response_id TEXT PRIMARY KEY,
            completion_rate REAL,
            average_rating REAL,
            quality_score REAL,
            quality_category TEXT,
            anomaly_status TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("Table created successfully!")


# 2. Insert Data
def insert_data():
    df = pd.read_csv("outputs/final_quality_results.csv")

    conn = sqlite3.connect(DATABASE)

    # Remove old records before inserting current data
    conn.execute("DELETE FROM survey_responses")

    for _, row in df.iterrows():
        conn.execute("""
            INSERT INTO survey_responses
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(row["Response_ID"]),
            row["Completion_Rate"],
            row["Average_Rating"],
            row["Quality_Score"],
            row["Quality_Category"],
            row["Anomaly_Status"]
        ))

    conn.commit()
    conn.close()

    print("Data inserted successfully!")
    print("Records inserted:", len(df))


# 3. Display All Records
def view_data():
    conn = sqlite3.connect(DATABASE)

    df = pd.read_sql_query(
        "SELECT * FROM survey_responses", conn
    )

    conn.close()

    print("\nAll Survey Responses:")
    print(df.to_string(index=False))
    print("\nTotal Records:", len(df))


# 4. Search a Response
def search_data(response_id):
    conn = sqlite3.connect(DATABASE)

    result = conn.execute("""
        SELECT * FROM survey_responses
        WHERE response_id = ?
    """, (response_id.strip(),)).fetchone()

    conn.close()

    if result:
        print("\nResponse Found!")
        print("Response ID:", result[0])
        print("Completion Rate:", result[1])
        print("Average Rating:", result[2])
        print("Quality Score:", result[3])
        print("Quality Category:", result[4])
        print("Anomaly Status:", result[5])
    else:
        print("\nResponse not found!")


# 5. Update Quality Category
def update_data(response_id, new_category):
    conn = sqlite3.connect(DATABASE)

    cursor = conn.execute("""
        UPDATE survey_responses
        SET quality_category = ?
        WHERE response_id = ?
    """, (new_category, response_id.strip()))

    conn.commit()
    conn.close()

    if cursor.rowcount > 0:
        print("Response updated successfully!")
    else:
        print("Response not found!")


# 6. Delete a Response
def delete_data(response_id):
    conn = sqlite3.connect(DATABASE)

    cursor = conn.execute("""
        DELETE FROM survey_responses
        WHERE response_id = ?
    """, (response_id.strip(),))

    conn.commit()
    conn.close()

    if cursor.rowcount > 0:
        print("Response deleted successfully!")
    else:
        print("Response not found!")


# Main Program
if __name__ == "__main__":

    create_table()
    insert_data()
    view_data()

    # Search for a response that exists in the CSV
    df = pd.read_csv("outputs/final_quality_results.csv")
    response_id = str(df.iloc[0]["Response_ID"])

    print("\nSearching for:", response_id)
    search_data(response_id)

    # Uncomment to test Update or Delete
    # update_data(response_id, "Medium Quality")
    # delete_data(response_id)
