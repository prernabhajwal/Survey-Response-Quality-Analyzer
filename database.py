import sqlite3
import pandas as pd
import os

# Database file name
DATABASE = "survey_database.db"


# 1. Connect to database and create table
def create_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
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

    print("Database table created successfully!")


# 2. Insert survey data into database
def insert_data():
    df = pd.read_csv("outputs/final_quality_results.csv")

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO survey_responses
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(row["Response_ID"]),
            float(row["Completion_Rate"]),
            float(row["Average_Rating"]),
            float(row["Quality_Score"]),
            str(row["Quality_Category"]),
            str(row["Anomaly_Status"])
        ))

    conn.commit()
    conn.close()

    print("Survey data inserted successfully!")


# 3. Display stored survey data
def view_data():
    conn = sqlite3.connect(DATABASE)

    df = pd.read_sql_query(
        "SELECT * FROM survey_responses", conn
    )

    conn.close()

    print("\nStored Survey Responses:")
    print(df.head(10).to_string(index=False))
    print("\nTotal Records:", len(df))


# 4. Search for a response using its ID
def search_data(response_id):
    conn = sqlite3.connect(DATABASE)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM survey_responses
        WHERE response_id = ?
    """, (response_id,))

    result = cursor.fetchone()

    conn.close()

    if result:
        print("\nResponse Found:")
        print(result)
    else:
        print("\nResponse not found!")


# Main program
if __name__ == "__main__":

    create_table()
    insert_data()
    view_data()

    # Example search
    search_data("SRQ0001")