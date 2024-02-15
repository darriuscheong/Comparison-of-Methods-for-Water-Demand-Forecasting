import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import os
import datetime
from os.path import dirname,abspath
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from pandas.plotting import autocorrelation_plot
from pmdarima.arima import auto_arima
from pmdarima.arima import ndiffs
from pmdarima.arima import nsdiffs
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

import time as time

import sys
sys.path.append(r'C:\Users\darri\OneDrive - Imperial College London\Darrius_Cheong_TimeSesriesAnalysis\04_Code\Python')

import glob_func
from glob_func import scores, split_output

#Import Data
import initialize
df = initialize.df
flow_wd = initialize.flow_wd
flow_weekend = initialize.flow_weekend

#Visualize Data
'''
plt.figure()
plt.plot(df['flow'].groupby(df['month']).mean())
plt.title('Monthly Mean Flow')
plt.figure()
plt.plot(df['flow'])
plt.figure()
sns.boxplot(x = 'month', y = 'flow', data=df)
plt.figure()
sns.boxplot(x = 'day', y = 'flow', data=df)
plt.figure()
plt.plot(flow_wd['datetime'],flow_wd['flow'],label='Average Workday Flow')
plt.plot(flow_weekend['datetime'],flow_weekend['flow'], label = 'Average non-workday flow')
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.legend()
'''

#Tests for stationarity and normality
'''
#Stationary Test
from statsmodels.tsa.stattools import adfuller
adf,pvalue,usedlag_,nobs,critical_values,icbest = adfuller(df['reduce'])
print("pvalue =", pvalue, "if pvalue > 0.05, data is not stationary")
#Results show stationary value

#Kolmogorow-Smirnov Test for Normality
from scipy.stats import kstest
kstest(df['flow'],'norm')
#Results are normally distributed

#QQ plot
import statsmodels.api as sm
import pylab as py
sm.qqplot(df['flow'],line='45')
py.show()
#Results show not normally distributed

from scipy import stats
fig = plt.figure()
ax1 = fig.add_subplot(211)
prob = stats.probplot(df['flow'], dist=stats.norm,plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probplot against normal distribution')

ax2 = fig.add_subplot(212)
flow2,_= stats.boxcox(df['flow'])
prob = stats.probplot(flow2, dist=stats.norm,plot=ax2)
ax2.set_title('Probplot after Box-Cox')
plt.show()

'''

#Differencing
'''
plt.figure(figsize = (12,9))
plt.subplot(6,1,1)
diff_flow = df['Reduced2']
autocorrelation_plot(diff_flow, label = '0 Order')
plt.legend(loc='upper left')
plt.xlim([0,2000])
plt.xticks(np.arange(0, 2000+1, 96))
for i in range(1,6):
  plt.subplot(6,1,i+1)
  diff_flow = diff_flow.diff(periods=96*7).fillna(0)
  autocorrelation_plot(diff_flow, label = '%3i Order' % i)
  plt.legend(loc='upper left')
  plt.xlim([0,2000])
  plt.xticks(np.arange(0, 2000+1, 96))
plt.suptitle('Autocorrelation for 1w differencing')
'''

#ACF & PACF
'''
#Autocorrelation for (flow)
plot_acf(df['flow'],lags = 96*8)
plot_pacf(df['flow'],lags = 96*8)

#Autocorrelation for (flow - diurnal flow)
plot_acf(df['reduced'],lags = 96*8)
plot_pacf(df['reduced'],lags = 96*8)
'''

#AutoArima - Diurnal flow as Exog
'''
arima_d = ndiffs(df['reduced'],max_d = 12)
arima_model = auto_arima(df['flow'], d = arima_d, 
                         exogenous = df['diurnal_flow'],
                         #trend = np.poly1d([1,1,1,1]),
                         seasonal=False, trace=True,
                         error_action = 'ignore',
                         suppress_warnings=True,stepwise= True,
                         n_fits = 50, method = 'nm')
print(arima_model.summary())
 
#Best model:  ARIMA(4,1,3)(0,0,0)[0] with exog
'''


#Split into train/test
size = int(len(df)*0.66)
X_train, X_test = df[:size],df[size:]

#ARIMAX (4,1,3) Model
start=time.time()
model_arimax = ARIMA(X_train['flow'],
                      order = (4,1,3), exog=X_train['diurnal_flow'],
                      freq = None
                    )
result_arimax=model_arimax.fit()
result_arimax.summary()
train_time_arimax = (time.time()-start)

#Train prediction
trainPredict_arimax = (result_arimax.predict(0,len(X_train)-1)).fillna(0)

#Test Prediction
forecast_arimax = result_arimax.get_prediction(len(X_train),len(df)-1,exog=X_test['diurnal_flow'])
testPredict_arimax = forecast_arimax.predicted_mean
testPredict_arimax_ci = forecast_arimax.conf_int(alpha=0.1)

#Evaluate Performance
testScore_arimax = scores(X_test['flow'],testPredict_arimax,testPredict_arimax_ci.iloc[:,0],testPredict_arimax_ci.iloc[:,1])
print('model_arimax',end=',')
for i in testScore_arimax.values:
    print(*i,end=',')
print(train_time_arimax)

#Save excel
excel_name = 'ARIMAX.xlsx'

df_results_test = pd.DataFrame({
            "testPredict":testPredict_arimax.tolist(),
            "testlower":testPredict_arimax_ci.iloc[:,0].tolist(),
            "testupper":testPredict_arimax_ci.iloc[:,1].tolist(),
            })

with pd.ExcelWriter(excel_name) as writer:
    df_results_test.to_excel(writer,index=False,sheet_name='Test')




#Deseasonalised ARIMA (2,1,2) Model
from statsmodels.tsa.arima.model import ARIMA
start=time.time()
model_arima = ARIMA(X_train['reduced'],
                      order = (2,1,2),
                      freq = None
                    )
result_arima=model_arima.fit()
result_arima.summary()
train_time_arima = (time.time()-start)

#Train prediction
trainPredict_arima = (result_arima.predict(0,len(X_train)-1)).fillna(0)+X_train['diurnal_flow']

#Test Prediction
forecast_arima = result_arima.get_prediction(len(X_train),len(df)-1)
testPredict_arima = forecast_arima.predicted_mean+X_test['diurnal_flow']
testPredict_arima_ci = forecast_arima.conf_int(alpha=0.1)
testPredict_arima_ci.iloc[:,0] = testPredict_arima_ci.iloc[:,0]+X_test['diurnal_flow']
testPredict_arima_ci.iloc[:,1] = testPredict_arima_ci.iloc[:,1]+X_test['diurnal_flow']

#ARIMA (4,1,3) Model
from statsmodels.tsa.arima.model import ARIMA
start=time.time()
model_arima2 = ARIMA(X_train['flow'],
                      order = (4,1,3),
                      freq = None
                    )
result_arima2=model_arima2.fit()
result_arima2.summary()
train_time_arima2 = (time.time()-start)

#Train prediction
trainPredict_arima2 = (result_arima2.predict(0,len(X_train)-1)).fillna(0)

#Test Prediction
forecast_arima2 = result_arima2.get_prediction(len(X_train),len(df)-1)
testPredict_arima2 = forecast_arima2.predicted_mean
testPredict_arima2_ci = forecast_arima2.conf_int(alpha=0.1)

#Evaluate Performance
testScore_arimax = scores(X_test['flow'],testPredict_arimax,testPredict_arimax_ci.iloc[:,0],testPredict_arimax_ci.iloc[:,1])
print('model_arimax',end=',')
for i in testScore_arimax.values:
    print(*i,end=',')
print(train_time_arimax)


testScore_arima = scores(X_test['flow'],testPredict_arima,testPredict_arima_ci.iloc[:,0],testPredict_arima_ci.iloc[:,1])
print('model_arima',end=',')
for i in testScore_arima.values:
    print(*i,end=',')
print(train_time_arima)

testScore_arima2 = scores(X_test['flow'],testPredict_arima2,testPredict_arima2_ci.iloc[:,0],testPredict_arima2_ci.iloc[:,1])
print('model_arima2',end=',')
for i in testScore_arima2.values:
    print(*i,end=',')
print(train_time_arima2)

trainScore_arimax = scores(X_train['flow'],trainPredict_arimax,0,0)
print('model_arimax',end=',')
for i in trainScore_arimax.values:
    print(*i,end=',')
print()
trainScore_arima = scores(X_train['flow'],trainPredict_arima,0,0)
print('model_arima',end=',')
for i in trainScore_arima.values:
    print(*i,end=',')

#Plot all
plt.figure()
plt.plot(X_test['flow'],label = 'original')
plt.plot(testPredict_arimax, label = 'ARIMAX(4,1,3)')
plt.plot(testPredict_arima, label = 'ARIMA(2,1,2)+Seasonal')
plt.plot(testPredict_arima2, label = 'ARIMA(4,1,3)')
plt.legend()


#Plotting ARIMAX residuals
testPredict = testPredict_arimax
testPredict_ci = testPredict_arimax_ci

testPredict_res = X_test['flow'] - testPredict
testPredict_lower_res = (testPredict - testPredict_ci.iloc[:,0])
testPredict_upper_res = (testPredict - testPredict_ci.iloc[:,1] )

plt.figure(figsize = (20,8))
plt.plot(testPredict_res, label = 'Residuals')
plt.fill_between(testPredict_res.index,testPredict_lower_res,testPredict_upper_res,alpha=0.1,color='b',label = 'Confidence')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = pd.to_datetime(labels)
ax.set_xticklabels(labels.day_of_week)
for i in range(0,len(labels)):
    dt = labels[i].strftime('%Y-%m-%d %H:%M:%S')
    if(X_test.index[0]<=labels[i]<=X_test.index[-1]):
      if(X_test.loc[dt]['holiday']):
        ax.get_xticklabels()[i].set_color("red")     
plt.xlabel('Day of week',fontsize=15)
plt.ylabel('Residuals ($m^3/15min$)',fontsize=15)
plt.legend(fontsize=18)
plt.suptitle('Residuals for ARIMAX(4,1,3) Model',fontsize=20)
plt.tight_layout()

#Plotting Deseasonalised ARIMA residuals
testPredict = testPredict_arima
testPredict_ci = testPredict_arima_ci

testPredict_res = X_test['flow'] - testPredict
testPredict_lower_res = (testPredict - testPredict_ci.iloc[:,0])
testPredict_upper_res = (testPredict - testPredict_ci.iloc[:,1] )

plt.figure(figsize = (20,8))
plt.plot(testPredict_res, label = 'Residuals')
plt.fill_between(testPredict_res.index,testPredict_lower_res,testPredict_upper_res,alpha=0.1,color='b',label = 'Confidence')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = pd.to_datetime(labels)
ax.set_xticklabels(labels.day_of_week)
for i in range(0,len(labels)):
    dt = labels[i].strftime('%Y-%m-%d %H:%M:%S')
    if(X_test.index[0]<=labels[i]<=X_test.index[-1]):
      if(X_test.loc[dt]['holiday']):
        ax.get_xticklabels()[i].set_color("red")     
plt.xlabel('Day of week',fontsize=15)
plt.ylabel('Residuals ($m^3/15min$)',fontsize=15)
plt.legend(fontsize=18)
plt.suptitle('Residuals for ARIMA(2,1,2) Model',fontsize=20)
plt.tight_layout()

#Autocorrelation for ARIMA Model
plt.figure()
autocorrelation_plot(testPredict)
plt.xlim([0,4000])
plt.xticks(np.arange(0, 4000+1, 96))
plt.title('Autocorrelation for ARIMA(4,1,3)')

'''===================================================================================='''

'''
#Autocorrelation for ARIMA Model
plt.figure()
autocorrelation_plot(testPredict)
plt.xlim([0,4000])
plt.xticks(np.arange(0, 4000+1, 96))
plt.title('Autocorrelation for ARIMA(4,1,3)')

#SARIMA Model
#Split into train/test
size = int(len(df)*0.66)
X_train, X_test = df[:size],df[size:]
AR_order = 96

from statsmodels.tsa.statespace.sarimax import SARIMAX
arima_model = SARIMAX(X_train['reduced'],
                      order = (2,0,2),
                      seasonal_order=(2,0,2,96*7))
result=arima_model.fit()
result.summary()

#Train prediction
train_prediction = result.predict(0,len(X_train)-1)+df['diurnal_flow']
prediction1 = result.predict(len(X_train),len(df)-1)+df['diurnal_flow']
forecast = result.get_prediction(len(X_train),len(df)-1,exog=X_test['diurnal_flow'])
prediction = forecast.predicted_mean
prediction_ci = forecast.conf_int()

#Plotting SARIMAX results
plt.figure()
prediction.plot(legend = True)
X_test['flow'].plot(legend = True)

import math
from sklearn.metrics import mean_squared_error
trainScore = math.sqrt(mean_squared_error(X_train['flow'], train_prediction))
print('Train Score : %.3f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(X_test['flow'],prediction))
print('Test Score : %.3f RMSE' % (testScore))

from statsmodels.tsa.arima.model import ARIMA
arima_model = ARIMA(X_train,
                      order = (AR_order,0,0)
                      #seasonal_order=(2,0,1,7))
                    )
result=arima_model.fit()
result.summary()

plt.figure()
autocorrelation_plot(result.predict(0,len(residual)))
plt.xlim([0,96*100])
plt.ylim([-0.25,0.25])
plt.title('Autocorrelation for AR(%d)' % AR_order)



#Autoarima of differenced flow
#Set arima m = 7 for weekly seasonality
arima_m = 96*7 

#Calculate d and D to speedup
arima_d = ndiffs(df['reduced'],max_d = 5)
arima_D = nsdiffs(df['reduced'], m = arima_m,max_D = 5)
max_lag = 12

arima_model = auto_arima(df['reduced'], start_p = 1, max_p =5,
                         start_q = 1, max_q = 5, d = arima_d,
                         m = arima_m, start_Q = 1,
                         max_Q = 5, start_P = 1, max_P = 5, 
                         D = arima_D, seasonal=True, trace=True,
                         error_action = 'ignore',
                         suppress_warnings=True,stepwise= True,
                         n_fits = 20,maxiter = 20, method = 'nm')
print(arima_model.summary())

#Extract daily seasonality (Diurnal Patterns)
d_decomposed = seasonal_decompose(df['flow'],model = 'additive',period = int(24*60/15))
d_trend = d_decomposed.trend.fillna(0)
d_seasonal = d_decomposed.seasonal.fillna(0)
d_residual = d_decomposed.resid.fillna(0)

#Extract weekly seasonality
decompfreq = int(24*60/15)*7
w_decomposed = seasonal_decompose(df['flow']-d_seasonal,model = 'additive',period = decompfreq)
trend = w_decomposed.trend.fillna(0)
w_seasonal = w_decomposed.seasonal.fillna(0)
residual = w_decomposed.resid.fillna(0)

#Daily Seasonality Plot
startd = 6
days = 7
endd = startd+days
i_start = startd*96
i_end = endd*96
plt.figure(figsize = (12,8))
plt.subplot(411)
plt.plot(d_seasonal[i_start:i_end], label ='Weekly Seasonality')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(d_trend[i_start:i_end],label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(d_seasonal[i_start:i_end],label='Daily Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(d_residual[i_start:i_end],label='Residual')
plt.legend(loc='upper left')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = pd.to_datetime(labels)
ax.set_xticklabels(labels.day_of_week)
plt.suptitle('Extracting Daily Seasonality')

#Weekly Seasonality Plot
startd = 6
days = 30
endd = startd+days
i_start = startd*96
i_end = endd*96
plt.figure(figsize = (12,8))
plt.subplot(411)
plt.plot(df['flow'][i_start:i_end], label ='Flow')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend[i_start:i_end],label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(w_seasonal[i_start:i_end],label='Weekly Seasonality')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = pd.to_datetime(labels)
ax.set_xticklabels(labels.day_of_week)
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual[i_start:i_end],label='Residual/White Noise')
plt.legend(loc='upper left')
plt.suptitle('Extracting Weekly Seasonality')

#Plot Overall Detrend
startd = 6
days = 30
endd = startd+days
i_start = startd*96
i_end = endd*96
plt.figure(figsize = (15,8))
plt.subplot(511)
plt.plot(df['flow'][i_start:i_end], label ='Flow')
plt.legend(loc='upper left')
plt.subplot(512)
plt.plot(d_seasonal[i_start:i_end],label='Daily Seasonality')
plt.legend(loc='upper left')
plt.subplot(513)
plt.plot(w_seasonal[i_start:i_end],label='Weekly Seasonality')
plt.legend(loc='upper left')
plt.subplot(514)
plt.plot(trend[i_start:i_end],label='Trend')
plt.legend(loc='upper left')
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
labels = [item.get_text() for item in ax.get_xticklabels()]
labels = pd.to_datetime(labels)
ax.set_xticklabels(labels.day_of_week)
plt.subplot(515)
plt.plot(residual[i_start:i_end],label='Residual')
plt.legend(loc='upper left')
plt.suptitle('Final Decomposition')

#Autocorrelation
plt.figure()
autocorrelation_plot(residual)
plt.ylim([-0.25,0.25])
plt.xlim([0,96*100])
plt.title('Autocorrelation for Residual')

#Set arima m = 7 for daily seasonality
arima_m = decompfreq

#Calculate d and D to speedup
arima_d = ndiffs(residual,max_d = 5)
arima_D = nsdiffs(residual, m = arima_m,max_D = 5)
max_lag = 96

arima_model = auto_arima(residual, start_p = 1, max_p =max_lag,
                         start_q = 1, max_q = max_lag, d = arima_d,
                         m = arima_m, start_Q = 1,
                         max_Q = 5, start_P = 1, max_P = 5, 
                         D = arima_D, seasonal=False, trace=True,
                         error_action = 'ignore',
                         suppress_warnings=True,stepwise= True,
                         n_fits = 50)
print(arima_model.summary())

# Best model for residual after seasonal decompose
# SARIMAX(5,0,0)

#Split into train/test
size = int(len(residual)*0.66)
X_train, X_test = residual[:size],residual[size:]
AR_order = 96

from statsmodels.tsa.arima.model import ARIMA
arima_model = ARIMA(X_train,
                      order = (AR_order,0,0)
                      #seasonal_order=(2,0,1,7))
                    )
result=arima_model.fit()
result.summary()

plt.figure()
autocorrelation_plot(result.predict(0,len(residual)))
plt.xlim([0,96*100])
plt.ylim([-0.25,0.25])
plt.title('Autocorrelation for AR(%d)' % AR_order)

#Train prediction
train_prediction = result.predict(0,len(X_train)-1) + w_seasonal[0:len(X_train)-1] \
                            + d_seasonal[0:len(X_train)-1] + trend[0:len(X_train)-1]
prediction = result.predict(len(X_train),len(residual)-1) + w_seasonal[len(X_train):len(residual)-1] \
                + d_seasonal[len(X_train):len(residual)-1] + trend[len(X_train):len(residual)-1]
X_train_flow = X_train + w_seasonal[0:len(X_train)-1] \
                + d_seasonal[0:len(X_train)-1] + trend[0:len(X_train)-1]

#Plotting SARIMAX results
plt.figure(figsize = (16,4))
prediction.plot(label = 'Prediction',legend = True)
X_train_flow.plot(label = 'Training',legend = True)

import math
from sklearn.metrics import mean_squared_error
trainScore = math.sqrt(mean_squared_error(X_train, train_prediction))
print('Train Score : %.3f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(X_test,prediction))
print('Test Score : %.3f RMSE' % (testScore))


#Initialize SARIMAX model
from statsmodels.tsa.statespace.sarimax import SARIMAX
arima_model = SARIMAX(X_train['flow'],
                      order = (2,0,0),
                      seasonal_order=(2,0,1,7))
result=arima_model.fit()
result.summary()

#Train prediction
train_prediction = result.predict(0,len(X_train)-1)
prediction = result.predict(len(X_train),len(df)-1)

#Plotting SARIMAX results
plt.figure()
prediction.plot(legend = True)
X_test['flow'].plot(legend = True)

import math
from sklearn.metrics import mean_squared_error
trainScore = math.sqrt(mean_squared_error(X_train['flow'], train_prediction))
print('Train Score : %.3f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(X_test['flow'],prediction))
print('Test Score : %.3f RMSE' % (testScore))
'''