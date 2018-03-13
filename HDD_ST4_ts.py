"""

This python file relates to building a time series model 
for the hard drives case. The idea is to focus on one 
model and serial number and see how the time series forecasts 
work on the different metrics of the website. 

"""

import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from statsmodels.tsa.stattools import adfuller
import seaborn as sns 
import fuckit 
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


from mods import howmanynulls, mmm 
#data = pd.read_csv("ST4000DM000.csv")

## Lets work with the serial number Z300X9WE

#data = data[data['serial_number']=="Z300X9WE"]
data = pd.read_csv("Z300X9WE.csv")

def test_stationary(timeseries):
    rolmean = pd.rolling_mean(timeseries, window=2)
    rolstd = pd.rolling_std(timeseries, window=2)

    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling mean')
    std = plt.plot(rolstd, color='black', label='Rolling std')

    plt.legend(loc='best')
    plt.title('Rolling mean and Stadard Deviation')
    plt.show(block=False)

    ## Perform the Dickey-Fuller test
    print('Results of Dickey-Fuller test')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                            index=['Test Statistic', 'p-value', 
                                    '#Lags used', 'Number of observations used'])

    for key,value in dftest[4].items():
        dfoutput['Critical value (%s)'%key] = value
    print(dfoutput)

smart_features_raw = [x for x in data.columns if "smart" in x and "raw" in x]
## From wiki, most important features
for i in [200, 201, 220, 222, 223, 224,225, 226, 250, 251,252, 254, 255, 195, 22, 11]:
    try:
        smart_features_raw.remove('smart_'+str(i)+'_raw')
    except ValueError: 
        print("Index not found: ", i)

## Find all the columns with a constant or zero values and drop NAs
col_names = []
data = data.dropna(axis=1)
for col in smart_features_raw:
    with fuckit: 
        if max(data[col]) == min(data[col]):
            col_names.append(col)

data = data.drop(col_names, axis=1)

## using only the columns that exist
smart_features_raw = [x for x in smart_features_raw if x in data.columns]
## Set the index to a time dependent problem
data = data.set_index(data.smart_9_raw/24.)


test_stationary(data.smart_190_raw)


