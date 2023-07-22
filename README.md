# This repository acts as a demonstration repository for Stock Info data processing and Machine Learning model development in Python


Data/Stock_info.csv a file generated from yahoo finance data to avoid fetching data every single time  
get_yfinance_data.py fetches yahoo finance data on all Nasdaq ticker companies   
DS-Sample.yml Conda environment used for the notebooks  
  
1. EDA.ipynb: jupyter notebook containing Explaratory Data Analysis (EDA) on NASDAQ company data obtained from yahoo finance API. Actions performed: Data encoding, reformatting, feature and their relationship analysis, Feature cross-correlation check.  Investigation if dataset is sufficient to perform overall Risk parameter prediction.  
  
2. ML-Baseline.ipynb We load "MSFT" stock data and try to build a Machine Learning model to predict if following day stock Closing price will be higher than the day before using Random Forest classifier. Classifier principal metrics are calculated (Recall, Precission, Accuracy, F1 score and confusion matrix). Procedure is repeated and same metrics are provided after feature engineering.  
  
3. ML-Sliding_window.ipynb We load "MSFT" stock data and perform feature engineering adding parameters such as (RSI, Standart moving average (SMA), Closing price ratios, trend indicators). Subsequently, the dataframe is reformed to contain parameters of several previous days (sliding window). The attempt to predict following day price change indicator using Random Forest classifier and Neural network is made. Classifier principal metrics (including RoC curve) are provided. Forward Feature Selection is used for the attempt to improve the results. As a following step, procedure is repeated, but this time the attempt is amde to predict the following day stock closing price and then converting it into price change indicator as a data post processing step (using Random Forest Regressor and Neural Network). Results are visualised, metrics calculated and compared. As a final note Principal Component Analysis is performed and the same predictions attempted (Closing price indicator and Closing price).  
  
4. ML-Decomposition.ipynb Not yet implemented: Company Stock info decomposition into seasonal variations and continued attempts into ML predictions  
  
5. ML-Multiple Companies.ipynb Not yet implemented: Continued ML predictions but with training on multiple company data  
  
6. Q learning.ipynb Not yet implemented: Reinforcement learning model for stock trading  


