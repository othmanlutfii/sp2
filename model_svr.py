import numpy as np
import math
import pandas as pd
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR


def prediksi_svr(df,lama_pred):
    closedf = df.filter(['Close'])

    scaler=MinMaxScaler(feature_range=(0,1))
    closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))
    training_size=int(len(closedf)*0.80)-30
    test_size=len(closedf)-training_size
    train_data,test_data=closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, time_step):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    time_step = 30
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)



    svr_rbf = SVR(kernel= 'rbf', C= 1e2, gamma= 0.1)
    svr_rbf.fit(X_train, y_train)

    # Lets Do the prediction 

    train_predict=svr_rbf.predict(X_train)
    test_predict=svr_rbf.predict(X_test)

    train_predict = train_predict.reshape(-1,1)
    test_predict = test_predict.reshape(-1,1)


    # Transform back to original form

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 
    # Step 6: Make predictions for the next 7 days
    last_sequence = test_data[-time_step:]  # Last sequence from the testing data

    predicted_prices = []

    harga_kedepan = []

    banyak_prediksi= lama_pred
    for _ in range(banyak_prediksi):
        next_price = svr_rbf.predict(last_sequence.reshape(1, time_step))
        predicted_prices.append(next_price)
        last_sequence = np.append(last_sequence[1:], next_price)

    
    predictions_svr = []
    for i in test_predict:
        pred_each= i[0]
        predictions_svr.append(pred_each)



    # Inverse transform the predicted prices to the original scale
    predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

    # Print the predicted prices for the next 7 days
    # print("Predicted Prices for the Next 7 Days:")
    # for price in predicted_prices:
    #     print(price[0])

    for price in predicted_prices:
        harga_pred = price[0]
        harga_kedepan.append(harga_pred)


    return(harga_kedepan,predictions_svr)