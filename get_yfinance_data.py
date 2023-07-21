# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing the yfinance package
import yfinance as yf
import pandas as pd
from datetime import datetime 
from dateutil.relativedelta import relativedelta 
from tqdm import tqdm

class stock_df_generator:
    def __init__(self):
        self._df_fname = 'Stock_info.csv'
        # list all stocktickers
        url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
        df=pd.read_csv(url, sep="|")
        self.tickers = list(df['Symbol'])[:-1]
        #self.tickers = ['AAPL', 'AAPU', 'KO', 'MSFT', 'AMZN']

    def set_dfname(self, name):
        """
        setter for stock info df file, only approve of .csv
        """
        if name[-4] == '.' and name[-3:] == 'csv': 
                self._df_fname = name
        else:
            self._df_fname = name[:-4]+'.csv'
        
    def get_dfname(self):
        return self._df_name
    
    def get_stock_info(self):
        try:
            df = pd.read_csv('Data/'+self._df_fname, low_memory=False, index_col=0, thousands=',')
            print('DF loaded')
        except FileNotFoundError:
            #Build a dictionary with information if file doesnt exists
            print('Building DF')
            periods = ['1', '6', '12'] #in months
            data = {}
            now = datetime.now()
            for ticker in tqdm(self.tickers):
                try:
                    the_stock = yf.Ticker(ticker)
                    data[ticker] = the_stock.info
                    for period in periods:
                        try:
                            hist = the_stock.history(period=period+'mo')
                            data[ticker][period+'mo_open_avg'] = hist['Open'].mean(axis=0)
                            data[ticker][period+'mo_close_avg'] = hist['Close'].mean(axis=0)
                            data[ticker][period+'mo_volume_avg'] = hist['Volume'].mean(axis=0)
                            data[ticker][period+'mo_low'] = hist['Low'].min(axis=0)
                            data[ticker][period+'mo_high'] = hist['High'].max(axis=0)
                        except:
                            pass
                    for interval_i in range(len(periods)-1):
                        t1 = now + relativedelta(months=-int(periods[interval_i]))
                        t2 = now + relativedelta(months=-int(periods[interval_i+1]))
                        t1 = t1.strftime('%Y-%m-%d')
                        t2 = t2.strftime('%Y-%m-%d')
                        hist = yf.download(ticker, t2, t1, progress=False)
                        data[ticker]['P'+str(interval_i)+'_open_avg'] = hist['Open'].mean(axis=0)
                        data[ticker]['P'+str(interval_i)+'_close_avg'] = hist['Close'].mean(axis=0)
                        data[ticker]['P'+str(interval_i)+'_volume_avg'] = hist['Volume'].mean(axis=0)
                        data[ticker]['P'+str(interval_i)+'_low'] = hist['Low'].min(axis=0)
                        data[ticker]['P'+str(interval_i)+'_high'] = hist['High'].max(axis=0)
                      
                except Exception as e:
                    print(ticker, str(e))
                    pass

            df = pd.DataFrame(data)
            self.save_df(df)
        return df
        
    def save_df(self, df):  
        df.to_csv(self._df_fname,index=True)
            

if __name__ == '__main__':
    df_generator = stock_df_generator()
    df = df_generator.get_stock_info()
    df_generator.save_df(df)