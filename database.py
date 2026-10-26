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


# 2. Save New Records
def save_results(df):
    create_table()
    conn = sqlite3.connect(DATABASE)

    for _, row in df.iterrows():
        response_id = str(row["Response_ID"]).strip()

        conn.execute("""
            INSERT OR IGNORE INTO survey_responses
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            response_id,
            row.get("Completion_Rate", 0),
            row.get("Average_Rating", 0),
            row["Quality_Score"],
            row["Quality_Category"],
            row.get("Anomaly_Status", "Normal")
        ))

    conn.commit()
    conn.close()


# 3. Retrieve All Records
def get_results():
    create_table()
    conn = sqlite3.connect(DATABASE)

    df = pd.read_sql_query(
        "SELECT * FROM survey_responses ORDER BY rowid DESC",
        conn
    )

    conn.close()
    return df


# 4. Display All Records
def view_data():
    df = get_results()
    print("\nAll Survey Records:")
    print(df.to_string(index=False))
    print("\nTotal Records:", len(df))


# 5. Search a Record
def search_data(response_id):
    conn = sqlite3.connect(DATABASE)

    result = conn.execute("""
        SELECT * FROM survey_responses
        WHERE response_id = ?
    """, (str(response_id).strip(),)).fetchone()

    conn.close()
    return result


# 6. Update Quality Category
def update_data(response_id, new_category):
    conn = sqlite3.connect(DATABASE)

    cursor = conn.execute("""
        UPDATE survey_responses
        SET quality_category = ?
        WHERE response_id = ?
    """, (new_category, str(response_id).strip()))

    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()

    return updated


# 7. Delete a Record
def delete_data(response_id):
    conn = sqlite3.connect(DATABASE)

    cursor = conn.execute("""
        DELETE FROM survey_responses
        WHERE response_id = ?
    """, (str(response_id).strip(),))

    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()

    return deleted


# 8. Test Database
if __name__ == "__main__":
    create_table()

    df = pd.read_csv("outputs/final_quality_results.csv")
    save_results(df)

    print("Data saved successfully!")
    view_data()

if __name__ == "__main__":
    create_table()

    conn = sqlite3.connect(DATABASE)
    conn.execute("DELETE FROM survey_responses")
    conn.commit()
    conn.close()

    df = pd.read_csv("outputs/final_quality_results.csv")
    save_results(df)

    print("Database reset and new data saved!")
    view_data()