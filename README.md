# Healthcare Social Listening Platform

### Voice of Patient & Healthcare Worker Intelligence System

A full end-to-end data pipeline that collects, processes, and visualizes
sentiment and topics from healthcare communities on Reddit — turning raw
social media text into actionable business insights.

---

## What this project does

Hospitals, insurers, and health tech companies spend millions trying to
understand what patients and healthcare workers actually think. This platform
automates that process:

- Collects posts from r/nursing, r/HealthInsurance, r/AskDocs
- Cleans and preprocesses raw text for NLP analysis
- Scores sentiment using VADER (rule-based)
- Clusters posts into topics using TF-IDF + KMeans
- Stores enriched data in a SQLite database
- Visualizes everything in a live interactive Streamlit dashboard

**Key finding from analysis:** r/HealthInsurance is the most negative
healthcare community (avg sentiment: -0.173), with "Insurance Denial &
Coverage" being the only topic with negative average sentiment (-0.169) —
driven by prior authorization delays, claim denials, and surprise billing.

---

## Dashboard Preview

- KPI cards: total posts, negative sentiment %, avg score, most negative community
- Sentiment breakdown donut chart
- Sentiment by subreddit bar chart
- Sentiment score distribution box plot
- Voice of Patient quotes (highest-signal negative posts)
- Topic analysis with color-coded sentiment bar chart
- Interactive topic explorer dropdown

---

## Tech Stack

| Layer              | Technology                     | Why                                   |
| ------------------ | ------------------------------ | ------------------------------------- |
| Data collection    | PRAW (Reddit API)              | Real-time subreddit scraping          |
| Data processing    | Python, Pandas, Regex          | Text cleaning and feature engineering |
| Sentiment analysis | VADER Sentiment                | Social media optimized NLP            |
| Topic modeling     | TF-IDF + KMeans (scikit-learn) | Unsupervised theme discovery          |
| Storage            | SQLite                         | Lightweight, portable database        |
| Visualization      | Streamlit + Plotly             | Interactive web dashboard             |
| Version control    | Git + GitHub                   | Professional workflow                 |

---

## Project Structure

healthcare-social-listening/
├── pipeline/
│ ├── scraper_mock.py # Data collection layer
│ ├── cleaner.py # Text preprocessing
│ ├── sentiment.py # VADER sentiment scoring
│ ├── topics.py # TF-IDF topic modeling
│ └── db.py # SQLite storage layer
├── dashboard/
│ └── app.py # Streamlit dashboard
├── data/ # SQLite database (gitignored)
├── reports/ # Executive briefs
├── requirements.txt
└── README.md

---

## How to run

```bash
# Clone the repo
git clone https://github.com/SaipriyaReddyTurpu/healthcare-social-listening.git
cd healthcare-social-listening

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python pipeline/db.py

# Launch the dashboard
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## Key Insights

1. **Insurance communities are most negative** — r/HealthInsurance scores
   -0.173 average sentiment vs +0.199 for r/AskDocs
2. **Insurance Denial is the only negative topic** — scoring -0.169, driven
   by prior authorization delays, claim denials, and surprise billing

3. **Nursing Burnout has the highest volume** — 16 posts (35% of dataset),
   but surprisingly neutral sentiment (+0.072), suggesting nurses cope
   through community support

4. **Care Access & Quality is most positive** — +0.163, driven by
   appreciation for good doctors and telehealth convenience

---

## Skills demonstrated

- End-to-end data pipeline architecture
- Natural Language Processing (NLP)
- Unsupervised machine learning (clustering)
- SQL database design and querying
- Interactive dashboard development
- API integration and data collection
- Business insight generation from unstructured data

---

## Future enhancements

- [ ] Live Reddit API integration (pending approval)
- [ ] HuggingFace transformer model for deeper sentiment nuance
- [ ] BERTopic for more sophisticated topic modeling
- [ ] Time-series sentiment tracking
- [ ] Export to PowerPoint executive brief
- [ ] Deploy to Streamlit Cloud (public URL)

---

_Built as part of a healthcare analytics portfolio project | Arizona State University_
