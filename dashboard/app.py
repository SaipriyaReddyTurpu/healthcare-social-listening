import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlite3
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

DB_PATH = "data/health_listening.db"

st.set_page_config(
    page_title="Healthcare Social Listening",
    page_icon="🏥",
    layout="wide"
)

@st.cache_data
def load_data():
    """
    Cache the data so the dashboard doesn't re-query
    the database on every interaction — makes it fast.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM posts", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    return df

def run_pipeline_if_needed():
    """Run the full pipeline if database doesn't exist yet"""
    if not os.path.exists(DB_PATH):
        with st.spinner("Running data pipeline for first time..."):
            sys.path.append('pipeline')
            from pipeline.scraper_mock import generate_sample_data
            from pipeline.cleaner import prepare_dataframe
            from pipeline.sentiment import score_dataframe
            from pipeline.db import save_to_db

            all_data = []
            for sub in ["nursing", "HealthInsurance", "AskDocs"]:
                df = generate_sample_data(sub, 15)
                all_data.append(df)

            combined = pd.concat(all_data, ignore_index=True)
            cleaned  = prepare_dataframe(combined)
            scored   = score_dataframe(cleaned)
            save_to_db(scored)

run_pipeline_if_needed()
df = load_data()

# ── Sidebar filters ──────────────────────────────────────
st.sidebar.title("Filters")
selected_subs = st.sidebar.multiselect(
    "Subreddits",
    options=df['subreddit'].unique().tolist(),
    default=df['subreddit'].unique().tolist()
)
selected_sentiment = st.sidebar.radio(
    "Sentiment",
    options=["All", "positive", "negative", "neutral"],
    index=0
)

# Apply filters
filtered = df[df['subreddit'].isin(selected_subs)]
if selected_sentiment != "All":
    filtered = filtered[filtered['vader_label'] == selected_sentiment]

# ── Header ───────────────────────────────────────────────
st.title("Healthcare Social Listening Dashboard")
st.caption("Voice of Patient & Healthcare Worker — Reddit Analysis")
st.divider()

# ── KPI Row ──────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

neg_pct = round((filtered['vader_label'] == 'negative').mean() * 100, 1)
avg_score = round(filtered['vader_compound'].mean(), 3)
most_neg_sub = df.groupby('subreddit')['vader_compound'].mean().idxmin()

k1.metric("Total posts", f"{len(filtered):,}")
k2.metric("Negative sentiment", f"{neg_pct}%")
k3.metric("Avg sentiment score", avg_score)
k4.metric("Most negative community", f"r/{most_neg_sub}")

st.divider()

# ── Row 1: Sentiment breakdown + by subreddit ────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment breakdown")
    sent_counts = filtered['vader_label'].value_counts().reset_index()
    sent_counts.columns = ['sentiment', 'count']
    color_map = {'positive': '#1D9E75', 'negative': '#D85A30', 'neutral': '#888780'}
    fig1 = px.pie(
        sent_counts, values='count', names='sentiment',
        color='sentiment', color_discrete_map=color_map,
        hole=0.4
    )
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    fig1.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Sentiment by subreddit")
    sub_sentiment = filtered.groupby(['subreddit', 'vader_label']).size().reset_index(name='count')
    fig2 = px.bar(
        sub_sentiment, x='subreddit', y='count', color='vader_label',
        color_discrete_map=color_map, barmode='group'
    )
    fig2.update_layout(margin=dict(t=20, b=20), legend_title="Sentiment")
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Sentiment score distribution ─────────────────
st.subheader("Sentiment score distribution by subreddit")
fig3 = px.box(
    filtered, x='subreddit', y='vader_compound', color='subreddit',
    points='all',
    labels={'vader_compound': 'Sentiment Score (-1 to +1)', 'subreddit': 'Community'}
)
fig3.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="neutral")
fig3.update_layout(showlegend=False, margin=dict(t=20, b=20))
st.plotly_chart(fig3, use_container_width=True)

# ── Row 3: Voice of Patient quotes ──────────────────────
st.subheader("Voice of Patient — most negative posts")
st.caption("These are the highest-signal negative posts — what patients and workers are really saying")

most_neg = filtered[filtered['vader_label'] == 'negative'].nsmallest(8, 'vader_compound')

for _, row in most_neg.iterrows():
    with st.container():
        score_color = "🔴" if row['vader_compound'] < -0.5 else "🟠"
        st.markdown(f"{score_color} **r/{row['subreddit']}** — score: `{row['vader_compound']}`")
        st.markdown(f"> {row['title']}")
        st.divider()

# ── Topic Analysis Section ───────────────────────────────
st.divider()
st.subheader("What are people talking about? — Topic analysis")
st.caption("Posts automatically grouped into themes using TF-IDF clustering")

sys.path.insert(0, 'pipeline')
from pipeline.topics import extract_topics_simple

@st.cache_data
def load_topics():
    all_data = []
    from pipeline.scraper_mock import generate_sample_data
    from pipeline.cleaner import prepare_dataframe
    from pipeline.sentiment import score_dataframe
    for sub in ["nursing", "HealthInsurance", "AskDocs"]:
        all_data.append(generate_sample_data(sub, 15))
    combined = pd.concat(all_data, ignore_index=True)
    cleaned  = prepare_dataframe(combined)
    scored   = score_dataframe(cleaned)
    topic_df, _, topic_names = extract_topics_simple(scored)
    return topic_df, topic_names

topic_df, topic_names = load_topics()

# Topic distribution bar chart
topic_counts = topic_df.groupby('topic_name').agg(
    posts=('topic_name', 'count'),
    avg_sentiment=('vader_compound', 'mean')
).reset_index()
topic_counts['avg_sentiment'] = topic_counts['avg_sentiment'].round(3)
topic_counts['color'] = topic_counts['avg_sentiment'].apply(
    lambda x: '#D85A30' if x < -0.05 else '#1D9E75' if x > 0.05 else '#888780'
)

col1, col2 = st.columns(2)

with col1:
    fig_topics = px.bar(
        topic_counts, x='posts', y='topic_name',
        orientation='h',
        color='avg_sentiment',
        color_continuous_scale=['#D85A30', '#888780', '#1D9E75'],
        color_continuous_midpoint=0,
        labels={'posts': 'Number of posts', 'topic_name': 'Topic',
                'avg_sentiment': 'Avg sentiment'}
    )
    fig_topics.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_topics, use_container_width=True)

with col2:
    st.markdown("**Sentiment by topic**")
    for _, row in topic_counts.sort_values('avg_sentiment').iterrows():
        icon = "🔴" if row['avg_sentiment'] < -0.05 else "🟢" if row['avg_sentiment'] > 0.05 else "⚪"
        st.markdown(f"{icon} **{row['topic_name']}** — {row['posts']} posts, score: `{row['avg_sentiment']}`")

# Sample posts per topic
st.markdown("**Sample posts by topic**")
selected_topic = st.selectbox(
    "Choose a topic to explore",
    options=topic_df['topic_name'].unique().tolist()
)
topic_sample = topic_df[topic_df['topic_name'] == selected_topic][['title', 'subreddit', 'vader_label', 'vader_compound']].head(5)
st.dataframe(topic_sample, use_container_width=True)