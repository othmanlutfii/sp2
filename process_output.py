def banding_list(a, b, c):
    com_list = []
    for i in range(len(a)):
        if (a[i] == "UP" and b[i] == "UP") :
            com_list.append("UP")
        elif (b[i] == "UP" and c[i] == "UP"):
            com_list.append("UP")
        elif (a[i] == "UP" and c[i] == "UP"):
            com_list.append("UP")
        elif (a[i] == "UP" and b[i] == "UP"):
            com_list.append("UP")
        elif (a[i] == "UP" and b[i] == "UP" and c[i] == "UP"):
            com_list.append("UP")
        else:
            com_list.append("DOWN")
    return(com_list)
    


def var_naik_turun(df,list_dep):
    list_dep = list_dep.copy()
    data_terahir = df['Close'][-1]
    list_dep.insert(0, data_terahir)

    trend_list = []  # List baru untuk menyimpan hasil perulangan

    for i in range(len(list_dep)-1):
        if list_dep[i+1] < list_dep[i]:
            trend_list.append("DOWN")
        elif list_dep[i+1] >= list_dep[i]:
            trend_list.append("UP")
        else:
            trend_list.append("no change")
    return(trend_list)

import pandas as pd
def df_tab(a,b,c):
    # Sample dictionary
    my_dict = {'Tanngal': a,
            'Predicted_Price': b,
            'Indicator':c}

    # Create dataframe from dictionary
    df = pd.DataFrame(my_dict)

    return(df)
    
from datetime import datetime, timedelta

def tanggal_kedepan(tanggal_terahir,banyak_hari):

    start_date = tanggal_terahir
    selected_days = banyak_hari

    # Convert the start date to a datetime object
    start_datetime = datetime.strptime(start_date, '%d-%m-%Y')

    # Create a list of dates with the selected number of days
    date_list = [start_datetime + timedelta(days=i) for i in range(selected_days)]

    # Format the dates as strings in the desired format
    formatted_dates = [date.strftime('%d-%m-%Y') for date in date_list]

    # Print the list of formatted dates

    return(formatted_dates)

def merge_final(a,e,b,c,d):    

    import pandas as pd

    # Example lists
    dates = a
    list_act = e
    list1 = b
    list2 = c
    list3 = d

    # Create a dictionary with lists as values
    data = {'Date': dates,'Nilai_asli': list_act, 'model_lstm': list1, 'model_arima': list2, 'model_svr': list3}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Convert 'Date' column to datetime type
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)
    return(df)


def vis_comp(df,var):
    # Visualize the data
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10,6))
    plt.title('Perbandingan hasil prediksi dengan nilai sebenarnya')
    plt.xlabel('Tanggal', fontsize=6)
    plt.ylabel('Harga Saham (Rp)', fontsize=6)
    plt.plot(df[['Nilai_asli',var]], linewidth=1)
    plt.legend(['Nilai_asli',var], loc='lower right')
    plt.savefig(f'static/output_files/{var}hasil.jpg')

def acc_model(a,b,c,d):
    from sklearn.metrics import mean_squared_error, r2_score
    mse_lstm = mean_squared_error(a, b)
    r2_lstm = r2_score(a, b)
    mse_arima = mean_squared_error(a, c)
    r2_arima = r2_score(a, c)
    mse_svr = mean_squared_error(a, d)
    r2_arima = r2_score(a, d)
    import pandas as pd

    # Create a dictionary with data
    data = {'Error': ['mse_lstm', 'r2_lstm', 'mse_arima', 'r2_arima','mse_svr','r2_arima'],
            'value': [mse_lstm, r2_lstm, mse_arima, r2_arima,mse_svr,r2_arima]}

    # Create the DataFrame
    df = pd.DataFrame(data)

    return (df)

