from sentence_transformers import SentenceTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')      # Run once, then works offline

model = SentenceTransformer('all-MiniLM-L6-v2')     # Downloads model locally
sid = SentimentIntensityAnalyzer()

def process_narratives(narrative_texts):
    narratives = {}
    for i, text in enumerate(narrative_texts):
        embedding = model.encode(text)
        sentiment = sid.polarity_scores(text)['compound']
        narratives[i] = {'text': text, 'embedding': embedding, 'sentiment': sentiment}
    return narratives