import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scraper_mock import generate_sample_data
from cleaner import prepare_dataframe

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text: str) -> dict:
    """
    VADER returns 4 scores:
    - compound: overall score from -1 (most negative) to +1 (most positive)
    - pos/neg/neu: proportion of text that is each sentiment
    
    We use compound to classify:
    >= 0.05  = positive
    <= -0.05 = negative
    in between = neutral
    
    Why 0.05 threshold? VADER's own recommendation for social media text.
    """
    scores = analyzer.polarity_scores(text)
    
    if scores['compound'] >= 0.05:
        label = 'positive'
    elif scores['compound'] <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    
    return {
        'vader_compound': round(scores['compound'], 4),
        'vader_positive': round(scores['pos'], 4),
        'vader_negative': round(scores['neg'], 4),
        'vader_neutral':  round(scores['neu'], 4),
        'vader_label':    label
    }

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print(f"Running VADER sentiment on {len(df)} posts...")
    
    vader_results = df['full_text'].apply(get_vader_sentiment)
    vader_df = pd.DataFrame(vader_results.tolist())
    df = pd.concat([df, vader_df], axis=1)
    
    return df

def print_summary(df: pd.DataFrame):
    print("\n--- Sentiment Summary ---")
    
    # Overall breakdown
    counts = df['vader_label'].value_counts()
    total = len(df)
    for label, count in counts.items():
        pct = round(count / total * 100, 1)
        bar = '█' * int(pct / 3)
        print(f"  {label:<10} {bar:<20} {pct}% ({count} posts)")
    
    # By subreddit
    print("\n--- Sentiment by Subreddit ---")
    for sub in df['subreddit'].unique():
        sub_df = df[df['subreddit'] == sub]
        neg_pct = round((sub_df['vader_label'] == 'negative').mean() * 100, 1)
        avg_compound = round(sub_df['vader_compound'].mean(), 3)
        print(f"  r/{sub:<20} neg: {neg_pct}%   avg score: {avg_compound}")
    
    # Most negative posts
    print("\n--- Top 3 Most Negative Posts ---")
    most_negative = df.nsmallest(3, 'vader_compound')
    for _, row in most_negative.iterrows():
        print(f"  [{row['vader_compound']}] r/{row['subreddit']}: {row['title'][:70]}...")

if __name__ == "__main__":
    # Load and clean data
    all_data = []
    for sub in ["nursing", "HealthInsurance", "AskDocs"]:
        df = generate_sample_data(sub, 15)
        all_data.append(df)
    
    combined = pd.concat(all_data, ignore_index=True)
    cleaned = prepare_dataframe(combined)
    
    # Score sentiment
    scored = score_dataframe(cleaned)
    
    # Show results
    print_summary(scored)
    
    print(f"\nColumns now: {list(scored.columns)}")
    print("\nSentiment analysis complete!")