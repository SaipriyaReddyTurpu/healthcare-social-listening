import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper_mock import generate_sample_data
from cleaner import prepare_dataframe

def extract_topics_simple(df: pd.DataFrame) -> pd.DataFrame:
    """
    We use sklearn's TF-IDF + KMeans clustering as our topic model.
    Why not BERTopic directly? BERTopic needs 50+ documents minimum
    to work well. With 45 posts we use this lighter approach that
    works identically on small datasets.
    
    TF-IDF = Term Frequency-Inverse Document Frequency
    It finds words that appear often in one group but rarely overall.
    That's how we identify what makes each topic unique.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import numpy as np

    texts = df['full_text'].tolist()
    
    print(f"Extracting topics from {len(texts)} posts...")
    
    # TF-IDF converts text to numbers
    # max_features=50: only look at top 50 most meaningful words
    # stop_words: ignore common words like 'the', 'and', 'is'
    vectorizer = TfidfVectorizer(
        max_features=50,
        stop_words='english',
        ngram_range=(1, 2)  # look at single words AND two-word phrases
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # KMeans groups similar posts into clusters = topics
    # n_clusters=5: we want 5 distinct topics
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['topic_id'] = kmeans.fit_predict(tfidf_matrix)
    
    # Find the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topic_labels = {}
    
    for topic_id in range(n_clusters):
        # Get the centroid (center) of each cluster
        center = kmeans.cluster_centers_[topic_id]
        # Top 5 words closest to the center = best description of topic
        top_indices = center.argsort()[-5:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topic_labels[topic_id] = " | ".join(top_words)
    
    df['topic_words'] = df['topic_id'].map(topic_labels)
    
    # Give each topic a human-readable name based on its keywords
    topic_names = {}
    for tid, words in topic_labels.items():
        words_lower = words.lower()
        if any(w in words_lower for w in ['insurance', 'denied', 'claim', 'coverage']):
            topic_names[tid] = "Insurance Denial & Coverage"
        elif any(w in words_lower for w in ['nurse', 'staff', 'hospital', 'patient ratio']):
            topic_names[tid] = "Nursing Burnout & Staffing"
        elif any(w in words_lower for w in ['cost', 'bill', 'afford', 'debt', 'price']):
            topic_names[tid] = "Healthcare Costs & Billing"
        elif any(w in words_lower for w in ['doctor', 'diagnosis', 'care', 'access']):
            topic_names[tid] = "Care Access & Quality"
        else:
            topic_names[tid] = f"Topic {tid}: {words[:30]}"
    
    df['topic_name'] = df['topic_id'].map(topic_names)
    
    return df, topic_labels, topic_names

def print_topic_summary(df: pd.DataFrame, topic_names: dict):
    print("\n--- Topic Summary ---")
    
    for tid, name in topic_names.items():
        topic_posts = df[df['topic_id'] == tid]
        count = len(topic_posts)
        avg_sentiment = round(topic_posts['vader_compound'].mean(), 3) if 'vader_compound' in df.columns else 'N/A'
        print(f"\n  Topic {tid}: {name}")
        print(f"  Posts: {count} | Avg sentiment: {avg_sentiment}")
        print(f"  Example: {topic_posts['title'].iloc[0][:70]}...")

if __name__ == "__main__":
    from sentiment import score_dataframe

    all_data = []
    for sub in ["nursing", "HealthInsurance", "AskDocs"]:
        df = generate_sample_data(sub, 15)
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    from cleaner import prepare_dataframe
    cleaned  = prepare_dataframe(combined)
    scored   = score_dataframe(cleaned)

    result_df, topic_labels, topic_names = extract_topics_simple(scored)
    print_topic_summary(result_df, topic_names)
    
    print(f"\nTopic distribution:")
    print(result_df['topic_name'].value_counts().to_string())
    print("\nTopic modeling complete!")