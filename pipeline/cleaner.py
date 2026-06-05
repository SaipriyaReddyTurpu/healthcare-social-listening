import re
import pandas as pd
from scraper_mock import generate_sample_data

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)           # remove URLs
    text = re.sub(r'\[deleted\]|\[removed\]', '', text)  # Reddit tombstones
    text = re.sub(r'[*_~`>#]', '', text)           # markdown symbols
    text = re.sub(r'\n+', ' ', text)               # newlines to spaces
    text = re.sub(r'\s+', ' ', text)               # collapse whitespace
    return text.strip().lower()

def build_full_text(row) -> str:
    """
    Combine title + body into one text blob.
    Why? Sentiment models perform better on longer context.
    """
    parts = [row.get('title', ''), row.get('selftext', '')]
    return " ".join([clean_text(p) for p in parts if p])

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Cleaning {len(df)} posts...")
    
    df['full_text'] = df.apply(build_full_text, axis=1)
    
    # Remove empty posts
    df = df[df['full_text'].str.len() > 10].reset_index(drop=True)
    
    # Convert timestamp to readable date
    from datetime import datetime
    df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    df['week'] = df['date'].dt.to_period('W').astype(str)
    df['month'] = df['date'].dt.to_period('M').astype(str)
    
    print(f"After cleaning: {len(df)} posts remaining")
    print(f"\nSample cleaned text:")
    print(f"  BEFORE: {df['title'].iloc[0]}")
    print(f"  AFTER:  {df['full_text'].iloc[0]}")
    
    return df

if __name__ == "__main__":
    # Load data from all 3 subreddits
    all_data = []
    for sub in ["nursing", "HealthInsurance", "AskDocs"]:
        df = generate_sample_data(sub, 15)
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Clean it
    cleaned = prepare_dataframe(combined)
    
    print(f"\nFinal dataset shape: {cleaned.shape}")
    print(f"Columns: {list(cleaned.columns)}")
    print("\nText cleaning complete!")