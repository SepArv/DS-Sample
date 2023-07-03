# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 12:55:30 2023

@author: arvyd
"""
import pandas as pd
from get_yfinance_data import stock_df_generator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data_getter = stock_df_generator()
raw_finance_data = data_getter.get_stock_info().transpose()

#print(raw_data.info())
#print(raw_data['overallRisk'])

raw_finance_data['overallRisk'] = pd.to_numeric(raw_finance_data['overallRisk'], errors='coerce', downcast = 'integer')

#Do some data encoding
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
raw_finance_data['country_label'] = le.fit_transform(raw_finance_data['country'])
raw_finance_data['industry_label'] = le.fit_transform(raw_finance_data['industry'])

"""Data exploration"""
print(raw_finance_data.info())
print(raw_finance_data.columns)
print(raw_finance_data.shape)

#create pie chart
'''
data = raw_finance_data['overallRisk'].dropna().value_counts()
df = pd.DataFrame({'Risk':data.index, 'Count':data.values})
ax = df.groupby(['Risk']).sum().plot(kind='pie', y='Count',autopct='%1.0f%%',
                                     legend=False, title = 'OverallRisk', ylabel = '')

'''

#lets analyse specific risk companies
#risk = [1.0, 2.0]
risk = [5.0, 10.0]
data = raw_finance_data.loc[raw_finance_data['overallRisk'].isin(risk)]
data = data.fillna(0)
data = data.apply(pd.to_numeric, errors='coerce', axis=1)

#do pairplot
params1 = ['debtToEquity','totalRevenue','totalDebt','totalCash','totalCashPerShare',
          'revenuePerShare','freeCashflow','operatingCashflow','recommendationMean']


sns.pairplot(data, vars = params1, hue =  'overallRisk',
             diag_kind="hist", palette = "tab10", kind = 'hist')
plt.show()


params2 = ['52WeekChange','enterpriseValue','profitMargins','fullTimeEmployees',
          'industry_label','country_label']

sns.pairplot(data, vars = params2, hue =  'overallRisk',
             diag_kind="hist", palette = "tab10", kind = 'hist')
plt.show()
pause
#lets look at particular cases
'''
import plotly.express as px
fig = px.scatter(data, x='totalRevenue',y='totalDebt', color = 'overallRisk',
                 hover_data=[data.index]
                 ,color_continuous_scale='bluered')
fig.show(renderer='browser')
'''

import matplotlib.pyplot as plt
import mplcursors   # separate package must be installed

# reproducible sample data as a pandas dataframe

fig, axs = plt.subplots(2,1, sharex=True)
axs[1].set_xlabel('totalRevenue')
colors = np.where(data['overallRisk'] <= risk[0], 'r', 'k')

axs[0].scatter(data['totalRevenue'], data['totalDebt'], c = colors)
axs[0].set_ylabel('totalDebt')
axs[1].scatter(data['totalRevenue'], data['freeCashflow'], c = colors)
axs[1].set_ylabel('freeCashflow')

#build legend
simart = [plt.scatter((0,1),(0,0), color= col, marker = 'o') for col in np.unique(colors)]
labels = [str(int(lbl)) for lbl in risk]
plt.legend(simart, labels, title= 'Overall Risk')
#make tooltips
for ax in axs:
    crs = mplcursors.cursor(ax,hover=True)
    crs.connect("add", lambda sel: sel.annotation.set_text(
        (data.index[sel.index])))
plt.show()


"""
-------------------PCA------------------------ 
"""

#perform rescaling
from sklearn.preprocessing import MinMaxScaler

data_for_pca = data[params1]#+params2]
#data_for_pca = data_for_pca.fillna(-1)

#scaler = MinMaxScaler()
#scaler.fit(data_for_pca)
#scaled = scaler.fit_transform(data_for_pca)
#scaled_data_for_pca = pd.DataFrame(scaled, columns=data_for_pca.columns)
scaled_data_for_pca=(data_for_pca-data_for_pca.mean())/data_for_pca.std()
scaled_data_for_pca = scaled_data_for_pca.fillna(-9)
########
from sklearn.decomposition import PCA
nr_components = 6

pca_finance = PCA(n_components=nr_components)
pca_finance_data = pca_finance.fit_transform(scaled_data_for_pca)

#create a DataFrame that will have the principal component values for all 569 samples.
pca_data_df = pd.DataFrame(data = pca_finance_data
             ,columns = ['principal component '+str(i+1) for i in range(nr_components)]
             ,index = data_for_pca.index)
print('PCA variation per principal component: {}'.format(pca_finance.explained_variance_ratio_))
print(sum(pca_finance.explained_variance_ratio_))

#plot the PCA's
for i in range(nr_components-1):
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component '+ str(i+1),fontsize=20)
    plt.ylabel('Principal Component '+ str(i+2),fontsize=20)
    plt.title("PCA of Yahoo finance data",fontsize=20)
    colors = ['r', 'g', 'b']
    #indicesToKeep = np.where(data['overallRisk'] >= risk[0], 1, 0)
    t0 = 0
    for target, color in zip(risk,colors):
        #indicesToKeep = data['overallRisk'] == target
        indicesToKeep = ((data['overallRisk'] > t0) & 
                                 (data['overallRisk'] <= target))
        t0 = target
        plt.scatter(pca_data_df.loc[indicesToKeep, 'principal component '+str(i+1)]                   ,pca_data_df.loc[indicesToKeep, 'principal component '+str(i+2)]
                   , c = color, s = 50, alpha = 0.5)
    
    plt.legend(risk,prop={'size': 15}, title = 'OverallRisk', title_fontsize= 14)
    plt.show()

'''
(LDA) (LDA) (LDA) 
'''


"""tSNE"""

"""Backwards elmination RFE"""












