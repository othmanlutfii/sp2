import numpy as np
import pandas as pd

from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
from datetime import datetime


def is_stock_available(comp):
    comp_with_jk = comp + ".JK"
    try:
        # Attempt to download stock data for the given symbol on IDX
        data = yf.download(comp_with_jk, period="1d")
        if data.empty:
            comp = comp
            return(comp)
        else:
            comp_with_jk = comp + ".JK"
            return(comp_with_jk)
    except:
         
        comp = ""
        return(comp)



def ambil_data(kode_saham):
    comp  = is_stock_available(kode_saham)
    if comp =="":

        print(comp)

        return(comp)
    else:
        print(comp)

        comp = comp.upper()
        end = datetime.now()
        start = datetime(end.year - 5, end.month, end.day)
        comp_dat = yf.download(comp, start, end)
        company_list = [comp_dat]
        company_name = [comp]

        for company, com_name in zip(company_list, company_name):
            company["company_name"] = com_name
            
        df = pd.concat(company_list, axis=0)


        # Assuming your dataframe is named 'df'
        dates = df.index
        dates = dates.tolist()

        # Assuming your list is named 'dates_list'
        list_date = [datetime.strftime(date, '%d-%m-%Y') for date in dates]
        return(df,list_date)

