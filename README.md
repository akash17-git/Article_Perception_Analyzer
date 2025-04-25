# Article Perception Analyzer

This project is designed to analyze articles based on their sentiment, emotion, topic, complexity, and other important factors. Users can either input the text directly or provide a URL to fetch the article's content. The system performs a series of analyses on the article, including sentiment analysis, emotion detection, topic classification, text summarization, and keyword extraction.

## Features

- **Sentiment Analysis**: Classifies the overall sentiment of the article (positive, negative, neutral).
- **Emotion Detection**: Identifies the primary emotion in the article (e.g., joy, anger, sadness, etc.).
- **Topic Classification**: Classifies the article into one of several predefined categories such as Sports, Politics, Technology, etc.
- **Text Summarization**: Summarizes the content of the article.
- **Named Entity Recognition (NER)**: Extracts named entities (e.g., people, locations, organizations).
- **Keyword Extraction**: Extracts important keywords from the article.
- **Word Complexity**: Measures the average syllables per word.
- **Average Sentence Length**: Calculates the average length of sentences in terms of words.

### How It Works

1. The user can input an article as direct text or provide a URL to fetch the article content.
2. The article is then processed using machine learning pipelines:
   - Sentiment and emotion analysis
   - Topic classification
   - Named entity recognition
   - Text summarization
3. Results are presented in a user-friendly table format with the following metrics:
   - Sentiment and emotion scores
   - Article summary
   - Named entities and keywords
   - Word complexity and average sentence length

### Requirements

To run this project, you need to install the required dependencies. Please use the `requirements.txt` file provided.
