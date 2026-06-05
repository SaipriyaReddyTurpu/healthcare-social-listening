import sqlite3
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper_mock import generate_sample_data
from cleaner import prepare_dataframe
from sentiment import score_dataframe

DB_PATH = "data/health_listening.db"

def save_to_db(df: pd.DataFrame, table_name: str = "posts"):
    """
    Saves enriched data to SQLite.
    if_exists='replace' means every run gives us a fresh clean table.
    """
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved {len(df)} rows to '{table_name}' table in {DB_PATH}")

def query_db(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

if __name__ == "__main__":
    # Build full pipeline
    print("Running full pipeline...")
    
    all_data = []
    for sub in ["nursing", "HealthInsurance", "AskDocs"]:
        df = generate_sample_data(sub, 15)
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    cleaned  = prepare_dataframe(combined)
    scored   = score_dataframe(cleaned)
    
    # Save to database
    save_to_db(scored)
    
    # Test querying it back
    print("\nTesting queries...")
    
    result = query_db("SELECT subreddit, COUNT(*) as posts, ROUND(AVG(vader_compound),3) as avg_sentiment FROM posts GROUP BY subreddit")
    print("\nSentiment by subreddit (from DB):")
    print(result.to_string(index=False))
    
    result2 = query_db("SELECT title, vader_compound, vader_label FROM posts ORDER BY vader_compound ASC LIMIT 5")
    print("\nTop 5 most negative posts (from DB):")
    for _, row in result2.iterrows():
        print(f"  [{row['vader_compound']}] {row['title'][:65]}...")
    
    print("\nDatabase layer complete!")