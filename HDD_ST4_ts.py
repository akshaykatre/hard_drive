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
from sksurv.linear_model import CoxPHSurvivalAnalysis

from mods import howmanynulls, mmm 
data = pd.read_csv("ST4000DM000.csv")

## Lets work with the serial number Z300X9WE

#data = data[data['serial_number']=="Z300X9WE"]
#data = pd.read_csv("Z300X9WE.csv")

def test_stationary(timeseries):
    rolmean = pd.rolling_mean(timeseries, window=2)
    rolstd = pd.rolling_std(timeseries, window=2)

    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling mean')
    std = plt.plot(rolstd, color='black', label='Rolling std')

    plt.legend(loc='best')
    plt.title('Rolling mean and Stadard Deviation')
  #  plt.show(block=False)

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
for i in [188, 200, 201, 220, 222, 223, 224,225, 226, 250, 251,252, 254, 255, 195, 22, 11]:
    try:
        smart_features_raw.remove('smart_'+str(i)+'_raw')
    except ValueError: 
        print("Index not found: ", i)


data = data.drop("Unnamed: 0", axis=1)
data.drop_duplicates(inplace=True)

## Find all the columns with a constant or zero values and drop NAs
col_names = []
data = data.dropna(axis=1)
for col in data.columns:
    with fuckit: 
        if max(data[col]) == min(data[col]):
            col_names.append(col)

data = data.drop(col_names, axis=1)

## using only the columns that exist
smart_features_raw = [x for x in smart_features_raw if x in data.columns]
## Set the index to a time dependent problem
#data = data.set_index(data.smart_9_raw/24.)

## Drop duplicate rows
#test_stationary(data.smart_190_raw)
## Group the dataset by serial numbers
grouped_df = data.groupby("serial_number")


''' 
The "problem" as such is that there are time-varying covariates which 
make it harder to implement into our survival models (at least really
simply and quickly). There are apparently ways to do it, but I need to 
read up on that 
'''
### What if I take all the values of the features and summarise them into one 
### row of entries - either taking their mean or so. That way I have one entry 
### for each hard-drive and will know if it passed or failed. 

## Make a new dataframe with the average of feature values 
## of each serial number

failed = data[data['failure']==1]
success = data[data['failure']==0]

## Make the mean of each feature in the grouped dataframe
data_x = grouped_df.mean()
## If the 'failure' feature > 1 then set the value of 
## failure to 1 - it means that the hard-drive has 
## definitely failed at the end 
data_x.loc[data_x['failure']> 0, 'failure'] = 1
## Move the index of grouped back as a column 
data_x.reset_index(level=0, inplace=True)
tuples_c = []
for x in data_x.iterrows():
    tuples_c.append((bool(x[1].failure), 
        (x[1].smart_9_raw/24.)))

data_y = np.array(tuples_c, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

## Now doing a simple kaplan_meier_estimator to see 
## the influence of features on the lifetimes

mask_low = data_x['smart_190_raw'] <= 25
mask_high = data_x['smart_190_raw'] > 25

time_treatment_low, survival_prob_treatment_low = kaplan_meier_estimator(
        data_y["Status"][mask_low],
        data_y["Survival_in_days"][mask_low])
plt.step(time_treatment_low, survival_prob_treatment_low, where="post",
             label="Temp <= 25")

time_treatment_high, survival_prob_treatment_high = kaplan_meier_estimator(
        data_y["Status"][mask_high],
        data_y["Survival_in_days"][mask_high])
plt.step(time_treatment_high, survival_prob_treatment_high, where="post",
             label="Temp > 25")

plt.show()

## Fixes for work machine - lets hope it works! 
data_numeric = data_x.select_dtypes(include=['category', 'float64'])

# data_numeric = data_x.select_dtypes('float64')
data_numeric = data_numeric.drop('failure', axis=1)

rows_to_convert = ['smart_240_raw', 'smart_241_raw',
                    'smart_242_raw', 'smart_7_raw', 
                ]
for row in rows_to_convert: 
    data_numeric[row] = np.log(data_numeric[row])


np.random.seed(42)
msk = np.random.rand(len(data_numeric)) < 0.8
train = data_numeric[msk]
test = data_numeric[~msk]
train_d = data_y[msk]
test_d = data_y[~msk]

estimator = CoxPHSurvivalAnalysis()

estimator.fit(train[smart_features_raw[:3]], train_d)
estimator.score(train[smart_features_raw[:3]], train_d)
estimator.score(test[smart_features_raw[:3]], test_d)




## Now the problems remain that I cannot use all the 
## columns that I want to use - and of course the prediction
## is quite bad because I only use a couple of rows

## The Cox model loses its gradient after a few iterations 
## need to have it fixed somehow! 