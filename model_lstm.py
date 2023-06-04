import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime


def prediksi_lstm(df,lama_pred):

    data = df.copy()
    # Create a new dataframe with only the 'Close column 
    data = data.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    # Get the number of rows to train the model on
    training_data_len = int(np.ceil( len(dataset) * .80 ))
    # Normalize the prices between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(dataset)

    # Split the data into training and testing sets
    train_size = int(len(scaled_prices) * 0.8) - 30
    train_data = scaled_prices[:train_size]
    test_data = scaled_prices[train_size:]

    # Step 2: Create sequences
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length-1):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    seq_length = 30  # Number of previous days' prices to use as input features
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mse',)

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Get the models predicted price values 
    predictions_lstm = []
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    for i in predictions:
        pred_each= i[0]
        predictions_lstm.append(pred_each)


    # Step 6: Make predictions for the next 7 days
    last_sequence = test_data[-seq_length:]  # Last sequence from the testing data
    predicted_prices = []
    harga_kedepan = []

    banyak_prediksi = lama_pred
    for _ in range(banyak_prediksi):
        next_price = model.predict(last_sequence.reshape(1, seq_length, 1))
        predicted_prices.append(next_price)
        last_sequence = np.append(last_sequence[1:], next_price)



    # Inverse transform the predicted prices to the original scale
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    # Print the predicted prices for the next 7 days
    for price in predicted_prices:
        harga_pred = price[0]
        harga_kedepan.append(harga_pred)


    actual_price = df['Close'][train_size:].tolist()

        

    return(harga_kedepan,predictions_lstm,actual_price)



