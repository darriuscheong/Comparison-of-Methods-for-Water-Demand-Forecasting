#This file extracts flow data and stores it in datafram df
#df contains 'flow', booleans for 'holiday' and 'workday', extracted 'Diurnal Seasonlity'
#Extracts diurnal patterns for weekday and nonweekdays as flow_wd and flow_weekend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
import os
import datetime
from os.path import dirname,abspath
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from pmdarima.arima import auto_arima
from pmdarima.arima import ndiffs
from pmdarima.arima import nsdiffs
from statsmodels.tsa.statespace.sarimax import SARIMAX

#Set path to data
#dir = dirname(dirname(dirname(abspath('initalize.py'))))
#datapath = os.path.join(dir,'02_Data','Flow Demand','data','flow_data.csv')
#datapath = r'C:\Users\darri\OneDrive - Imperial College London\Darrius_Cheong_TimeSesriesAnalysis\02_Data\Flow Demand\data\flow_data.csv'
#Initialising Data
df = pd.read_csv('flow_data.csv')
df['date'] = pd.to_datetime(df['date'],dayfirst=True)
df['datetime'] = df['date']
df['year'] = [d.year for d in df.datetime]
df['month'] = [d.month for d in df.datetime]
df['day'] = [d.dayofweek for d in df.datetime]
df['date'] = [d.date() for d in df.datetime]
df['time'] = [d.time() for d in df.datetime]
df.set_index('datetime',inplace=True)
#df.flow.replace(to_replace = 0,value = 1e-10, inplace =True)

#UK Holiday Calender
from workalendar.europe import UnitedKingdom
cal = UnitedKingdom()
holidays = set(holiday[0]
               for year in range(df['year'].min(),df['year'].max()+1)
               for holiday in cal.holidays(year)
               )
df['holiday'] = df['date'].isin(holidays)
df['workday'] = np.where((df['holiday']==True) | (df['day']>=5), False, True )

#Average flow for workdays
df2 = df['flow'].loc[df['workday']==True]
flow_wd = df2.groupby(df2.index.time).mean().to_frame()
flow_wd['datetime'] = [datetime.datetime.combine(df['date'][0], t) for t in flow_wd.index]

#Average flow of weekend/holidays
df3 = df['flow'].loc[df['workday']==False]
flow_weekend = df3.groupby(df3.index.time).mean().to_frame()
flow_weekend['datetime'] = [datetime.datetime.combine(df['date'][0], t) for t in flow_weekend.index]

#Create workday flow and holiday flow 
df['workday_flow']  = pd.concat([(flow_wd['flow'])]*int(len(df)/len(flow_wd)),ignore_index=True).values
df['holiday_flow']  = pd.concat([(flow_weekend['flow'])]*int(len(df)/len(flow_wd)),ignore_index=True).values

#Create diurnal flow absed on workday and holiday flows
df['diurnal_flow'] = df['workday_flow']
df.loc[df['workday']==False,'diurnal_flow'] = df.loc[df['workday']==False,'holiday_flow']

#Backfill missing data based on workday and holiday flows
df.loc[df['flow']<1e-3,'flow'] = df.loc[df['flow']<1e-3,'diurnal_flow']

#Vector for days
day_arr = []
for i in range(0,len(df['flow'])):
    day_vec = np.array([0,0,0,0,0,0,0])
    day_vec[df['day'].iloc[i]] = 1
    day_arr.append(day_vec)
df['day_arr']=day_arr

#Subtract seasonality
df['reduced'] = df['flow']-df['diurnal_flow']

'''
#R2 test
from sklearn.metrics import r2_score
R2_wd = r2_score(df['flow'].loc[df['workday']==True],df['workday_flow'].loc[df['workday']==True])
R2_nwd = r2_score(df['flow'].loc[df['workday']==False],df['holiday_flow'].loc[df['workday']==False])
print('R2 score for workday: %.3f' % R2_wd)
print('R2 score for non-workday: %.3f' % R2_nwd)

#Ftest
from scipy.stats import f_oneway
print('F-test for workday flow:')
f_oneway(df['flow'].loc[df['workday']==True],df['workday_flow'].loc[df['workday']==True])

print('F-test for non-workday flow:')
f_oneway(df['flow'].loc[df['workday']==False],df['holiday_flow'].loc[df['workday']==False])

'''

#Check Distribution
'''
from scipy.stats import norm, weibull_min, gamma, gumbel_r

data = df['flow']

#Histogram
plt.hist(data,bins = 50, label = 'Flow', density=True)

#Normal Dist
mu,std = norm.fit(data)   
xmin,xmax = plt.xlim()
x = np.linspace(xmin,xmax,100)
p_norm = norm.pdf(x,mu,std)
plt.plot(x,p_norm,'k', linewidth=2, label='Norm')

#Gumbel_r
loc, scale = gumbel_r.fit(data)
p_gumbel_r = gumbel_r.pdf(x,loc,scale)
plt.plot(x,p_gumbel_r,'r', linewidth=2, label='gumbel_r')

#Weibull
c, loc, scale = weibull_min.fit(data)
p_weibull_min = weibull_min.pdf(x,c,loc,scale)
plt.plot(x,p_weibull_min,'g', linewidth=2, label='weibull_min')

#Gamma
a, loc, scale = gamma.fit(data)
p_gamma = gamma.pdf(x,a,loc,scale)
plt.plot(x,p_gamma,'m', linewidth=2, label='gamma')

plt.legend()
plt.title("Distribution of flow")

'''

'''==========================================================================================================='''

'''
plt.figure(figsize = (24,24))
plt.subplots(3,2)
for hour in range(0,23,4):
    plt.subplot(3,2,int(hour/4)+1)
    data = df['flow'].loc[(df['time'] == datetime.time(hour=hour)) & (df['workday'] ==True)]
    
    #Histogram
    plt.hist(data,bins = 50, label = 'Distribution for %i hour' % hour, density=True)

    #Normal Dist
    mu,std = norm.fit(data)   
    xmin,xmax = plt.xlim()
    x = np.linspace(xmin,xmax,100)
    p_norm = norm.pdf(x,mu,std)
    plt.plot(x,p_norm,'k', linewidth=2, label='Norm')

    #Gumbel_r
    loc, scale = gumbel_r.fit(data)
    p_gumbel_r = gumbel_r.pdf(x,loc,scale)
    plt.plot(x,p_gumbel_r,'r', linewidth=2, label='gumbel_r')

    #Weibull
    c, loc, scale = weibull_min.fit(data)
    p_weibull_min = weibull_min.pdf(x,c,loc,scale)
    plt.plot(x,p_weibull_min,'g', linewidth=2, label='weibull_min')

    #Gamma
    a, loc, scale = gamma.fit(data)
    p_gamma = gamma.pdf(x,a,loc,scale)
    plt.plot(x,p_gamma,'m', linewidth=2, label='gamma')

    plt.legend(fontsize = "5")
    plt.xticks(fontsize = "5")
    plt.yticks(fontsize = "5")
    #plt.tight_layout()
plt.suptitle('Distributions across different times (Non Workday)')

for hour in range(0,24,1):
    for minute in range(0,60,15):
        data = df['flow'].loc[(df['time'] == datetime.time(hour=hour,minute=minute)) & (df['workday'] ==True)]

'''

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


'''============================================================================================='''

#Daily Seasonality Plot
'''
startd = 6
days = 28
endd = startd+days
i_start = startd*96
i_end = endd*96
plt.figure(figsize = (12,8))
plt.subplot(311)
plt.plot(df['flow'][i_start:i_end], label ='Flow')
plt.legend(loc='upper left')
plt.subplot(312)
plt.plot(df['Diurnal Seasonality'][i_start:i_end],label='Diurnal Seasonality')
plt.legend(loc='upper left')
plt.subplot(313)
plt.plot(df['Reduced'][i_start:i_end],label='Diurnal Seasonality Removed')
plt.legend(loc='upper left')
plt.suptitle('Subtracting Dirunal Seasonality')

#Autocorrelation
plt.figure()
autocorrelation_plot(df['Reduced'])
#plt.ylim([-0.25,0.25])
plt.xlim([0,96*10])
plt.title('Autocorrelation for Flow - Diurnal')

#Extract weekly seasonality
decompfreq = int(24*60/15)*7
decomposed = seasonal_decompose(df['reduced'],model = 'additive',period = decompfreq)
trend = decomposed.trend.fillna(0)
seasonal = decomposed.seasonal.fillna(0)
residual = decomposed.resid.fillna(0)

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
plt.plot(df['Diurnal Seasonality'][i_start:i_end],label='Diurnal Seasonality')
plt.legend(loc='upper left')
plt.subplot(513)
plt.plot(seasonal[i_start:i_end],label='Weekly Seasonality')
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

#Differencing
plt.figure()
autocorrelation_plot(df['Reduced'].diff(periods=96).fillna(0))
plt.title('Autocorrelation after differencing of 1 d')
plt.figure()
autocorrelation_plot(df['Reduced'].diff(periods=96*7).fillna(0))
plt.title('Autocorrelation after differencing of 1 w')
plt.figure()
autocorrelation_plot(df['Reduced'].diff(periods=96*7*4).fillna(0))
plt.title('Autocorrelation after differencing of 1 m')


#Tests for stationarity and normality

#Stationary Test
from statsmodels.tsa.stattools import adfuller
adf,pvalue,usedlag_,nobs,critical_values,icbest = adfuller(df['flow'])
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

df['flow']=flow2
'''
