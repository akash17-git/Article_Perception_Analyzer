from transformers import pipeline
from textstat import textstat
import nltk
import re
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from rich.console import Console
from rich.table import Table
import os
import logging
import transformers
import requests
from bs4 import BeautifulSoup
nltk.download('punkt_tab')

# --- SETUP ---
nltk.download('punkt', quiet=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

# --- PIPELINES ---
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
emotion_analyzer = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", return_all_scores=True)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
ner_tagger = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# --- FUNCTION TO GET ARTICLE TEXT FROM URL ---
# --- FUNCTION TO CHECK IF THE INPUT IS A URL ---
def is_url(text):
    # Simple check to see if the input starts with 'http' or 'www'
    url_pattern = re.compile(r"^(https?://|www\.)[^\s]+$")
    return bool(url_pattern.match(text))

# --- FUNCTION TO GET ARTICLE INPUT (AUTO-IDENTIFY) ---
def get_article_input():
    article_text = input("Please enter the article text or URL: ")
    
    # Check if the input is a URL
    if is_url(article_text):
        print("URL detected. Fetching the article content...")
        article_text = fetch_article_from_url(article_text)  # Fetch the article from the URL
        return article_text
    else:
        print("Direct text detected. Proceeding with analysis...")
        return article_text
def fetch_article_from_url(url):
    try:
        # Fetch the content from the URL
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the article text (you can modify this based on the website's structure)
        paragraphs = soup.find_all('p')
        article_text = " ".join([para.get_text() for para in paragraphs if para.get_text()])
        
        return article_text
    except Exception as e:
        return f"Error fetching article: {e}"

# --- ANALYSIS FUNCTIONS ---
def classify_topic(text):
    candidate_labels = ["Sports", "Politics", "Technology", "Health", "Business", "Entertainment", "Science", "Education"]
    result = topic_classifier(text, candidate_labels=candidate_labels)
    return result['labels'][0], result['scores'][0]  # Return the most probable topic
  
def analyze_sentiment(text):
    return sentiment_analyzer(text[:512])[0]

def analyze_emotion(text):
    emotions = emotion_analyzer(text[:512])[0]
    emotions = sorted(emotions, key=lambda x: x['score'], reverse=True)
    top = emotions[0]
    return top if top['score'] > 0.5 else {"label": "neutral", "score": top['score']}

def summarize_text(text):
    max_len = 512
    if len(text.split()) > max_len:
        text = " ".join(text.split()[:max_len])
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

def calculate_word_complexity(text):
    words = word_tokenize(text)
    if len(words) == 0:
        return 0
    return round(sum([textstat.syllable_count(w) for w in words]) / len(words), 2)

def average_sentence_length(text):
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        return 0
    return round(sum(len(word_tokenize(s)) for s in sentences) / len(sentences), 2)

def extract_entities(text):
    entities = ner_tagger(text[:512])
    return sorted(set(ent["word"] for ent in entities))

def extract_keywords(text, num_keywords=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform([text])
    scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
    sorted_keywords = sorted(scores, key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_keywords[:num_keywords]]

def generate_article_overview(sentiment, emotion):
    return f"This article carries a {sentiment.lower()} sentiment and an emotional undertone of {emotion.lower()}."

# --- MASTER FUNCTION ---
# --- MASTER FUNCTION WITH TOPIC CLASSIFICATION ---
def analyze_article(text):
    sentiment = analyze_sentiment(text)
    emotion = analyze_emotion(text)
    summary = summarize_text(text)
    complexity = calculate_word_complexity(text)
    avg_sentence_len = average_sentence_length(text)
    entities = extract_entities(text)
    keywords = extract_keywords(text)
    overview = generate_article_overview(sentiment['label'], emotion['label'])
    topic, topic_score = classify_topic(text)  # Get topic classification

    return {
        "Sentiment": sentiment['label'],
        "Sentiment Score": round(sentiment['score'], 2),
        "Emotion": emotion['label'].upper(),
        "Emotion Score": round(emotion['score'], 2),
        "Topic": topic,
        "Topic Score": round(topic_score, 2),
        "Summary": summary,
        "Word Complexity (Syllables/Word)": complexity,
        "Average Sentence Length (Words)": avg_sentence_len,
        "Named Entities": entities,
        "Keywords": keywords,
        "Article Overview": overview
    }


# --- PRETTY PRINT ---
import json

def safe_print(value):
    try:
        # Attempt to print the value directly
        print(value)
    except UnicodeEncodeError:
        # Handle any encoding error by converting problematic characters to their unicode representation
        print(str(value).encode('utf-8', 'replace').decode('utf-8'))

def display_results(result):
    console = Console()
    table = Table(title="📊 Article Analysis Results")

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key in [
        "Sentiment", "Sentiment Score",
        "Emotion", "Emotion Score",
        "Topic", "Topic Score",  # Added topic and topic score
        "Word Complexity (Syllables/Word)", "Average Sentence Length (Words)",
        "Article Overview"]:
        table.add_row(key, str(result[key]))

    console.print(table)

    console.rule("📝 Summary")
    safe_print(result["Summary"])  # Use safe_print for Summary output

    console.rule("💬 Named Entities")
    safe_print(", ".join(result["Named Entities"]))  # Use safe_print for Named Entities output

    console.rule("📌 Keywords")
    safe_print(", ".join(result["Keywords"]))  # Use safe_print for Keywords output


# --- USAGE ---
article_text = get_article_input()

# Check if article_text is not empty
if article_text:
    result = analyze_article(article_text)
    display_results(result)
else:
    print("No valid article text entered.")
