import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
from nltk.corpus import cmudict
nltk.download('cmudict')
from collections import Counter
from flask import Flask, send_file
import pandas as pd


# Function to download stop words from a text file on Google Drive
file_ids = ['1mBOuggD8AVNFjr9sprLoD2_6mVWAgRGE', '1RKxMOHzBdLrGuYb7MCJRTKKPwDG9Agbe',
            '1PnZhcsfjBVxnzwa4N6MrLWf6Kuhhjpdk', '1tTDfLXNPxNuUGZXHQkQhW6wPf4Xnivwr',
            '13LXnH6vaJhvY4s2ai_2oW2qwongU_iAI', '1K-6MjPq5AQg4ICYY6PDfapB7JECUnryD',
            '1aWxyJI0d9MOk59OZ_unfBY5E-Nvg_ezW']

def download_stop_words_folder(file_id, encoding='utf-8', errors='ignore'):
    url = f'https://drive.google.com/uc?export=download&id={file_id}'
    response = urlopen(url)
    stop_words_folder = [line.decode(encoding, errors=errors).strip() for line in response.readlines()]
    return set(stop_words_folder)

# Download and combine stop words from all text files
all_stop_words = set()
for file_id in file_ids:
    stop_words_folder = download_stop_words_folder(file_id)
    all_stop_words.update(stop_words_folder)

# Function to remove stop words from the given text
def remove_stopwords(text, stop_words_folder):
    return ' '.join(word for word in text.split() if word.lower() not in stop_words_folder)



# Function to download a file from Google Drive
def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?id=" + file_id
    response = requests.get(URL)
    with open(destination, 'wb') as f:
        f.write(response.content)

# Google Drive file IDs for positive and negative words
positive_words_id = '1seAj8G42SmfgUUx8lqVDJofm4Tuh2TOT'
negative_words_id = '1qqMwc_-ayS38HEOB97osO_nkIxRkbnvh'

# File paths to save positive and negative words lists
positive_words_path = 'positive_words.txt'
negative_words_path = 'negative_words.txt'

# Download positive and negative words lists
download_file_from_google_drive(positive_words_id, positive_words_path)
download_file_from_google_drive(negative_words_id, negative_words_path)


# Function to read the content from a text file
def read_text_file(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()

# Read the content from the positive and negative words files
positive_words_content = read_text_file(positive_words_path, encoding='ISO-8859-1')
negative_words_content = read_text_file(negative_words_path, encoding='ISO-8859-1')


# Tokenize positive and negative words
positive_words = re.findall(r'\b\w+\b', positive_words_content)
negative_words = re.findall(r'\b\w+\b', negative_words_content)


# Calculate Scores
def score(tokens, filtered_positive_words, filtered_negative_words):
    positive_score = sum(1 for word in tokens if word in filtered_positive_words)
    negative_score = -1 * sum(1 for word in tokens if word in filtered_negative_words)
    denominator = positive_score + negative_score + 0.000001
    polarity_score = (positive_score - negative_score) / denominator
    subjectivity_score = (positive_score + negative_score) / (len(tokens) + 0.000001)
    print("Positive Score:", positive_score)
    print("Negative Score:", negative_score)
    print("Polarity Score:", polarity_score)
    print("Subjectivity Score:", subjectivity_score)

# Calculate average sentence length
def calculate_average_sentence_length(tokens):
    sentences = nltk.sent_tokenize(' '.join(tokens))
    return len(tokens) / len(sentences)

# Function to identify percentage of complex words (words with more than two syllables)
def identify_complex_words(tokens):
    vowels = 'aeiou'
    count = sum(1 for word in tokens if count_syllables(word) > 2)
    percentage_of_complex_words = (count / len(tokens)) * 100
    return count, percentage_of_complex_words

# Function to calculate Fog index
def calculate_fog_index(average_sentence_length, percentage_of_complex_words):
    fog_index = 0.4 * (average_sentence_length + percentage_of_complex_words)
    return fog_index

# Function to calculate Average Number of Words Per Sentence
def calculate_average_words_per_sentence(tokens):
    sentences = nltk.sent_tokenize(' '.join(tokens))
    average_words_per_sentence = len(tokens) / len(sentences)
    return average_words_per_sentence

# Function to count syllables in a word, handling exceptions
def count_syllables(word):
    # Count vowels, excluding consecutive vowels
    vowels = 'aeiou'
    count = sum(1 for char in word if char in vowels and not (char == 'e' and word.count('e') > 1))

    # Handle special case of single 'e' at the end (not counted as a syllable)
    if word.endswith('e') and count > 1:
        count -= 1

    return max(1, count)

# Function to count personal pronouns in the text
personal_pronouns = ['I', 'we', 'my', 'ours', 'us']
def count_personal_pronouns(text):
    # Use regex to find occurrences of personal pronouns
    pronoun_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(pronoun) for pronoun in personal_pronouns) + r')\b', re.IGNORECASE)
    matches = pronoun_pattern.findall(text)
    return len(matches)

# Function to identify average word length
def average_word_length(tokens):
    total_characters = sum(len(word) for word in tokens)
    total_words = len(tokens)
    average_word_length = total_characters / total_words
    print("Average word length :  ", average_word_length)
    return average_word_length


def analyze_url_readability(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').text.strip()

        article_text = ''
        elements = soup.find_all(['p', 'ol'])
        exclusion_class = 'tdm-descr'

        for element in elements:
            if exclusion_class not in element.get('class', []):
                if element.name == 'p':
                    text = ' '.join(element.text.split())
                    article_text += text + '\n'
                elif element.name in ['ol']:
                    list_items = element.find_all('li')
                    for item in list_items:
                        text = ' '.join(item.text.split())
                        article_text += f'{text}\n'

        combined_text = f"{title}\n\n{article_text}"

        tokens = word_tokenize(combined_text)
        filtered_positive_words = set(positive_words)
        filtered_negative_words = set(negative_words)

        positive_score = sum(1 for word in tokens if word in filtered_positive_words)
        negative_score = sum(1 for word in tokens if word in filtered_negative_words)
        denominator = positive_score + negative_score + 0.000001
        polarity_score = round((positive_score - negative_score) / denominator)
        subjectivity_score = round((positive_score + negative_score) / (len(tokens) + 0.000001))

        sentences = nltk.sent_tokenize(combined_text)
        average_sentence_length = round(len(tokens) / len(sentences))

        complex_word_count = sum(1 for word in tokens if count_syllables(word) > 2)
        percentage_of_complex_words = round(100 * (complex_word_count / len(tokens)) )

        fog_index = round(0.4 * (average_sentence_length + percentage_of_complex_words))

        personal_pronoun_count = count_personal_pronouns(combined_text)

        total_characters = sum(len(word) for word in tokens)
        total_words = len(tokens)
        average_word_length = round(total_characters / total_words)

        article_type = 'POSITIVE' if positive_score > negative_score else 'NEGATIVE'

        result = {
            'Type': article_type,
            'Title': title,
            'Positive Score': positive_score,
            'Negative Score': negative_score,
            'Polarity Score': polarity_score,
            'Subjectivity Score': subjectivity_score,
            'Average number of Words Per Sentence': average_sentence_length,
            'Percentage of Complex Words': percentage_of_complex_words,
            'Fog Index': fog_index,
            'Personal Pronoun Count': personal_pronoun_count,
            'Average Word Length': average_word_length
        }

        print("Readability analysis result:")
        for key, value in result.items():
            print(f"{key}: {value}")

        return result
    else:
        print("Failed to retrieve the page. Status code:", response.status_code)
        return None


#url = input("Enter the URL to analyze: ")
#result = analyze_url_readability(url)
