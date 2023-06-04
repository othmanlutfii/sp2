import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from flask import Flask, render_template, request
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')


def filter_zip(*args):
    return zip(*args)

def clean_text(text):
    if text is not None and isinstance(text, str):
        if isinstance(text, str):
            text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
            text = re.sub(r'#', '', text)  # Remove hashtags
            text = re.sub(r'RT[\s]+', '', text)  # Remove retweets
            text = re.sub(r'https?:\/\/\S+', '', text)  # Remove hyperlinks
            text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
            text = re.sub(r'^\s+|\s+?$', '', text)  # Remove leading/trailing spaces
            text = text.lower()  # Convert to lowercase
        else:
            text = ''
    else:
        text = ''
    return text

def get_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)

    if sentiment_scores['compound'] > 0:
        return 'Positive'
    elif sentiment_scores['compound'] < 0:
        return 'Negative'
    else:
        return 'Neutral'
    
def analyze_sentiment():
    # Perform sentiment analysis
    symbol = request.args.get('symbol')
    today = datetime.today().date()
    since_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")

    # Tentukan kueri pencarian
    query = f'${symbol} since:{since_date}'

    # Ambil tweet dan simpan dalam sebuah list
    tweets = []
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        tweets.append([tweet.date, tweet.rawContent])

    # Konversi list ke DataFrame dan simpan sebagai file CSV
    df = pd.DataFrame(tweets, columns=['date', 'rawContent'])

    cleaned_tweets = [clean_text(tweet) for tweet in df['rawContent'] if tweet is not None]
    sentiment_labels = [get_sentiment(tweet) for tweet in cleaned_tweets if tweet is not None]

    positive_count = sentiment_labels.count('Positive')
    negative_count = sentiment_labels.count('Negative')
    neutral_count = sentiment_labels.count('Neutral')

    # Get the last 5 tweets and their sentiment labels
    last_tweets = df.tail(5)['rawContent'].tolist()
    last_sentiments = sentiment_labels[-5:]

    # Create a DataFrame with the last 5 tweets and their sentiment labels
    last_tweets_df = pd.DataFrame({'Tweet': last_tweets, 'Sentiment': last_sentiments})

    labels = ['Positive', 'Neutral', 'Negative']
    values = [positive_count, neutral_count, negative_count]

    # Visualize sentiment distribution
    plt.bar(labels, values)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title(f'Stock {symbol} Sentiment Analysis')


     # Add value labels to the bars
    for i, value in enumerate(values):
        plt.text(i, value, str(value), ha='center')

    # Save the plot to an image file
    plot_file = f'static/plot_{symbol}.png'
    plt.savefig(plot_file)