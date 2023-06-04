import pandas as pd
import numpy as np
from flask import json
import scrap_data
import model_lstm
import model_svr
import model_arima
import process_output
import snscrape.modules.twitter as sntwitter
import re
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from flask import Flask, render_template, request
import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from datetime import datetime, timedelta



import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

yf.pdr_override()

# For time stamps
from datetime import datetime


app = Flask(__name__)




@app.route('/')
def index():
    return render_template("index.html")


@app.route('/dashboard', methods=['POST','GET'])
def submit(): 

    #take imput data
    name = request.form['namasaham']
    banyakprediksi = request.form['banyakprediksi']

    # namasaham = request.args.get('namasaham') 
    # banyakprediksi = request.args.get('banyakprediksi')
    # name = namasaham
    # banyakprediksi = request.form['banyakprediksi']
    
    banyakprediksi=int(banyakprediksi)
    banyakprediksi1 = str(banyakprediksi)

    #scrap dataset
    data_saham = scrap_data.ambil_data(name)
    date_data = data_saham[1]
    data_saham = data_saham[0]

    #doing prediction
    prediksi1= model_lstm.prediksi_lstm(data_saham,banyakprediksi)
    prediksi2= model_svr.prediksi_svr(data_saham,banyakprediksi)
    prediksi3 = model_arima.prediksi_arima(data_saham,banyakprediksi)


    
    #take value each prediction
    pred_dep_lstm = prediksi1[0]
    pred_dep_lstm = [round(num) for num in pred_dep_lstm]

    pred_lstm = prediksi1[1]
    pred_lstm = [round(num) for num in pred_lstm]

    actual_price = prediksi1[2]

    pred_dep_arima = prediksi2[0]
    pred_dep_arima = [round(num) for num in pred_dep_arima]

    pred_arima = prediksi2[1]
    pred_arima = [round(num) for num in pred_arima]

    pred_dep_svr = prediksi3[0]
    pred_dep_svr = [round(num) for num in pred_dep_svr]

    pred_svr = prediksi3[1]
    pred_svr = [round(num) for num in pred_svr]


    #make number
    numbers_pred = []
    for i in range(1, banyakprediksi+1):
        numbers_pred.append(i)

    number_act = []
    for i in range(1, int(len(actual_price))+1):
        number_act.append(i)

    today_date = date_data[-1]



    list_tanggal_prediksi = process_output.tanggal_kedepan(today_date,banyakprediksi)

    test_data_len = len(pred_svr)
    list_data_test = date_data[-test_data_len:]
    actual_price_dat = actual_price[-test_data_len:]

    print(len(list_data_test))
    print(len(actual_price_dat))
    print(name)
    print(len(pred_lstm))
    print(len(pred_arima))
    print(len(pred_svr))

    df_gab_pred = process_output.merge_final(list_data_test,actual_price_dat,pred_lstm,pred_arima,pred_svr)

    process_output.vis_comp(df_gab_pred,'model_lstm')
    process_output.vis_comp(df_gab_pred,'model_arima')
    process_output.vis_comp(df_gab_pred,'model_svr')




    
    #tabel lstm
    output_tabel_lstm = process_output.var_naik_turun(data_saham,pred_dep_lstm)
    df_tabel_lstm = process_output.df_tab(list_tanggal_prediksi,pred_dep_lstm,output_tabel_lstm)
    df_tabel_lstm = df_tabel_lstm.to_dict('records')

    #tabel arima
    output_tabel_arima = process_output.var_naik_turun(data_saham,pred_dep_arima)
    df_tabel_arima = process_output.df_tab(list_tanggal_prediksi,pred_dep_arima,output_tabel_arima)
    df_tabel_arima = df_tabel_arima.to_dict('records')


    #tabel svr
    output_tabel_svr = process_output.var_naik_turun(data_saham,pred_dep_svr)
    df_tabel_svr= process_output.df_tab(list_tanggal_prediksi,pred_dep_svr,output_tabel_svr)
    df_tabel_svr = df_tabel_svr.to_dict('records')



    #convert data
    pred_dep_lstm = [float(item) for item in pred_dep_lstm]
    pred_dep_arima = [float(item) for item in pred_dep_arima]
    pred_dep_svr = [float(item) for item in pred_dep_svr]
    pred_lstm = [float(item) for item in pred_lstm]
    pred_dep_arima = [float(item) for item in pred_dep_arima]
    pred_svr = [float(item) for item in pred_svr]
    actual_price = [float(item) for item in actual_price]





    # print(prediksi1[0])
    # print(prediksi1[1])

    # print(prediksi2[0])
    # print(prediksi2[1])

    # print(prediksi3[0])
    # print(prediksi3[1])

    # print(len(prediksi1[0]))
    # print(len(prediksi1[1]))

    # print(len(prediksi2[0]))
    # print(len(prediksi2[1]))

    # print(len(prediksi3[0]))
    # print(len(prediksi3[1]))


    






    return render_template("dashboard.html", pred_dep_lstm=pred_dep_lstm,
                           name = name,
                           list_data_test = list_data_test,
                           pred_lstm = pred_lstm,
                           pred_dep_arima = pred_dep_arima,
                           pred_arima = pred_arima,
                           pred_dep_svr = pred_dep_svr,
                           pred_svr = pred_svr,
                           numbers = numbers_pred,
                           actual_price = actual_price,
                           number_act = number_act,
                           banyakprediksi1 = banyakprediksi1 ,
                           kode_saham = name,
                           df_tabel_lstm = df_tabel_lstm,
                           df_tabel_arima = df_tabel_arima,
                           df_tabel_svr = df_tabel_svr)





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


@app.route('/analyze', methods=['POST', 'GET'])
def analyze_sentiment():
    # Perform sentiment analysis

    # name = request.form['namasaham']

    symbol = request.args.get('name')
    print(symbol)


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
    print(positive_count, neutral_count, negative_count)

    plt.bar(labels, values)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title(f'Stock {symbol} Sentiment Analysis')





    # Visualize sentiment distribution
    # plt.bar(df['Label Codes'], df['Values'])
    # plt.xlabel('Sentiment')
    # plt.ylabel('Count')
    # plt.title(f'Stock {symbol} Sentiment Analysis')


     # Add value labels to the bars
    for i, value in enumerate(values):
        plt.text(i, value, str(value), ha='center')

    # Save the plot to an image file
    plot_file = f'static/plot_{symbol}.png'
    plt.savefig(plot_file)
    return render_template('notifications.html', symbol=symbol, last_tweets=last_tweets_df.to_html(index=False), labels=labels, values=values, plot=plot_file)


if __name__ == "__main__":
    app.run(debug=True)
