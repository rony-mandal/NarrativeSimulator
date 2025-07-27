from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import pandas as pd

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

model = SentenceTransformer('all-MiniLM-L6-v2')
sid = SentimentIntensityAnalyzer()

def process_narratives(narrative_texts):
    narratives = {}
    for i, text in enumerate(narrative_texts):
        embedding = model.encode(text)
        sentiment = sid.polarity_scores(text)['compound']
        narratives[i] = {'text': text, 'embedding': embedding, 'sentiment': sentiment}
    return narratives

def load_narrative_data(file_path='data/psyops_narratives.csv'):
    df = pd.read_csv(file_path)
    narratives = {}
    for i, row in df.iterrows():
        text = row['text']
        sentiment = row.get('sentiment', sid.polarity_scores(text)['compound'])
        embedding = model.encode(text)
        narratives[i] = {'text': text, 'embedding': embedding, 'sentiment': sentiment}
    return narratives