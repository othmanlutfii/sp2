import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import lag_plot
import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error




def prediksi_arima(df,lama_pred):
    train_data, test_data = df[0:int(len(df)*0.8)-1], df[int(len(df)*0.8)+1:]

    train_ar = train_data['Close'].values
    test_ar = test_data['Close'].values
    from statsmodels.tsa.arima.model import ARIMA

    # https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
    history = [x for x in train_ar]
    print(type(history))
    predictions = list()
    for t in range(len(test_ar)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_ar[t]
        history.append(obs)
    predictions = np.array(predictions)
    harga_arima = model_fit.forecast(steps=lama_pred)
    harga_arima = harga_arima.tolist()

    predictions_arima = []

    for i in predictions:
        predictions_arima.append(i)


    return(harga_arima,predictions_arima)
