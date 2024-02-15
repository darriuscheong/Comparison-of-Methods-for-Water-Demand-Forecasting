import math
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import time as time
from scipy.stats import nbinom, norm

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from keras.models import Model, Sequential
from keras import backend as K

#RMSE
def rmse(test,pred):
    return math.sqrt(mse(test,pred))

#Nash Sutcliffe Efficiency
def nse(test,pred):
    return (1-(np.sum((test-pred)**2)/np.sum((test-np.mean(test))**2)))

#Interval Score for 90% confidence
def interval_score(test,lower,upper):
    ci = 0.9
    func1 = 2/(1-ci)
    int_score = np.mean((upper-lower)+func1*((lower-test)**(test<lower).astype(int)+(test-upper)**(test>upper).astype(int)))
    return(int_score)

#Evaluate all
def scores(test,pred,lower,upper):
    score_rmse = rmse(test,pred)
    score_mape = mape(test,pred)
    score_mae = mae(test,pred)
    score_nse = nse(test,pred)
    score_is = interval_score(test,lower,upper)
    df = pd.DataFrame({'indicator':['RMSE','MAPE','MAE','NSE','IS'],
          'value':[score_rmse,score_mape,score_mae,score_nse,score_is]})
    df.set_index('indicator',inplace=True)
    return (df)

def get_time_array(df):
    time_arr = df.index
    
    x = []
    #n_steps_in = n_steps_present + n_steps_recent + n_steps_distant
    for i in range(len(df)):
        if i+15+96 > len(df):
            #Prediction out of sequence
            break
        
        if ((i-96+int(15/2)<0)| (i-96*4+int(15/2)<0)):
        #Past data not available
            continue

        range_predict = range(i+15,i+15+96)

        output = time_arr[range_predict]

        x.append(output)
    return np.squeeze(np.array(x))

def split_sequence(sequence, n_steps_present, n_steps_recent, n_steps_distant, n_steps_out):
    x, y = [],[]
    #n_steps_in = n_steps_present + n_steps_recent + n_steps_distant
    for i in range(len(sequence)):
        if i+n_steps_present+n_steps_out > len(sequence):
            #Prediction out of sequence
            break
        
        if ((i-96+int(n_steps_recent/2)<0)| (i-96*4+int(n_steps_distant/2)<0)):
        #Past data not available
            continue

        range_present = range(i,i+n_steps_present)
        range_recent = range(i,i+n_steps_recent)
        range_distant = range(i,i+n_steps_distant)

        range_predict = range(i+n_steps_present,i+n_steps_present+n_steps_out)
        
        input_present = sequence[range_present] #Data 5 timesteps before
        input_recent =  sequence[[r-96+int(n_steps_recent/2) for r in range_recent]] #Data one day before
        input_distant =  sequence[[r-96*4+int(n_steps_distant/2) for r in range_distant]] #Data one week before

        input = np.concatenate((input_present, input_recent, input_distant),axis=None)
        output = sequence[range_predict]
        x.append(input)
        y.append(output)
    return np.squeeze(np.array(x)), np.squeeze(np.array(y))

def split_sequence_test(sequence, day, n_steps_present, n_steps_recent, n_steps_distant, n_steps_out):
    x, y = [],[]
    #n_steps_in = n_steps_present + n_steps_recent + n_steps_distant
    for i in range(len(sequence)):
        if i+n_steps_present+n_steps_out > len(sequence):
            #Prediction out of sequence
            break
        
        if ((i-96*day+int(n_steps_recent/2)<0)|(i-96*4+int(n_steps_recent/2)<0)):
            #Past data not available
            continue
            #input_recent = np.zeros(5)
        
        range_present = range(i,i+n_steps_present)
        range_recent = range(i,i+n_steps_recent)
        range_distant = range(i,i+n_steps_distant)

        range_predict = range(i+n_steps_present,i+n_steps_present+n_steps_out)
        
        input_present = sequence[range_present] #Data 5 timesteps before
        input_recent =  sequence[[r-96*4+int(n_steps_recent/2) for r in range_recent]] #Data one day before
        input_distant =  sequence[[r-96*day+int(n_steps_distant/2) for r in range_distant]] #Data two days before

        input = np.concatenate((input_present, input_recent, input_distant),axis=None)
        output = sequence[range_predict]
        x.append(input)
        y.append(output)
    return np.squeeze(np.array(x)), np.squeeze(np.array(y))

def split_sequence_exog(df, n_steps_present, n_steps_recent, n_steps_distant, n_steps_out):
    sequence = df['flow'].values.astype('float32')
    dow = df['day_arr'].values
    #dow = df['day'].values.astype('int')/6
    holiday = df['holiday'].values.astype('int')
    time = df.index.strftime("%H%M").astype('int')/2400
    
    x, y = [],[]
    #n_steps_in = n_steps_present + n_steps_recent + n_steps_distant
    for i in range(len(sequence)):
        if i+n_steps_present+n_steps_out > len(sequence):
            #Prediction out of sequence
            break
        
        if ((i-96+int(n_steps_recent/2)<0)| (i-96*4+int(n_steps_distant/2)<0)):
        #Past data not available
            continue

        range_present = range(i,i+n_steps_present)
        range_recent = range(i,i+n_steps_recent)
        range_distant = range(i,i+n_steps_distant)

        range_predict = range(i+n_steps_present,i+n_steps_present+n_steps_out)
        
        r = range_present
        input_present = sequence[r] #Data 5 timesteps before
        input_present_e = np.concatenate((dow[i],holiday[i],time[i]),axis=None)
        #input_present_e = np.concatenate((np.concatenate(dow[r]),holiday[r],time[r]),axis=None)

        r = [r-96+int(n_steps_recent/2) for r in range_recent]
        input_recent =  sequence[r] #Data one day before
        input_recent_e = np.concatenate((np.concatenate(dow[r]),holiday[r],time[r]),axis=None)

        r = [r-96*4+int(n_steps_distant/2) for r in range_distant]
        input_distant =  sequence[r] #Data 4days before
        input_distant_e = np.concatenate((np.concatenate(dow[r]),holiday[r],time[r]),axis=None)

        input = np.concatenate((input_present, input_recent, input_distant),axis=None)
        input = np.concatenate((input, input_present_e),axis=None)
        #input = np.concatenate((input, input_present_e, input_recent_e, input_distant_e),axis=None)
        output = sequence[range_predict]

        x.append(input)
        y.append(output)
    return np.squeeze(np.array(x)), np.squeeze(np.array(y))

def split_exog(df):
    sequence = df['flow'].values.astype('float32')
    #workday_flow = df['workday_flow'].values.astype('float32')
    #holiday_flow = df['holiday_flow'].values.astype('float32')
    dow = df['day'].values.astype('int')
    holiday = df['holiday'].values.astype('int')
    time = df.index.strftime("%H%M").astype('int')
    
    x = []
    #n_steps_in = n_steps_present + n_steps_recent + n_steps_distant
    for i in range(len(sequence)):
        if i+15+96 > len(sequence):
            #Prediction out of sequence
            break
        
        if ((i-96+int(15/2)<0)| (i-96*4+int(15/2)<0)):
        #Past data not available
            continue

        #input_wd = workday_flow[[r-7 for r in range_recent]]
        #input_hf = holiday_flow[[r-7 for r in range_recent]]
        input_dow = [0,0,0,0,0,0,0]
        input_dow[dow[i]] = 1
        input_holiday = [holiday[i],holiday[i+96]]
        input_time = time[i]/2400

        input=[]
        input = np.concatenate((input_dow,input_holiday,input_time),axis=None)
        x.append(input)

    #return np.squeeze(np.array(x))
    return (np.array(x))

def split_output(sequence):
    x = []
    #n_steps_in = n_steps_present + n_steps_recent + n_steps_distant
    for i in range(len(sequence)):
        if i+96 > len(sequence):
            #Prediction out of sequence
            break

        range_out = range(i,i+96)

        output = sequence[range_out]
        x.append(output)
    return np.squeeze(np.array(x))

def model_scores(model,X_train,X_test,Y_train,Y_test,sc):

    #Evaluate Test Data
    pred_params = model.predict(X_test,verbose=0)
    testPredict = norm.mean(pred_params[:,:96],pred_params[:,96:])
    lower, upper = norm.interval(0.9,pred_params[:,:96],pred_params[:,96:])

    #Evaluate Train Data
    pred_params = model.predict(X_train,verbose=0)
    trainPredict = norm.mean(pred_params[:,:96],pred_params[:,96:])
    train_lower, train_upper = norm.interval(0.9,pred_params[:,:96],pred_params[:,96:])

    #De-standardize data
    testPredict = sc.inverse_transform(testPredict)
    lower = sc.inverse_transform(lower)
    upper = sc.inverse_transform(upper)

    trainPredict = sc.inverse_transform(trainPredict)
    train_lower = sc.inverse_transform(train_lower)
    train_upper = sc.inverse_transform(train_upper)

    #Evaluate Performance
    trainScore = scores(sc.inverse_transform(Y_train),trainPredict,train_lower,train_upper)
    testScore = scores(sc.inverse_transform(Y_test),testPredict,lower,upper)
    
    '''
    print('Train Scores:')
    print(trainScore.transpose())
    print('Test Scores:')
    print(testScore.transpose())
    print('Train Time:')
    print(train_time)
    '''

    score_arr = [trainScore['value'][0], trainScore['value'][1],trainScore['value'][2],trainScore['value'][3],trainScore['value'][4],
                 testScore['value'][0], testScore['value'][1],testScore['value'][2],testScore['value'][3],testScore['value'][4]]

    return score_arr

def model_config(model_name,df):
    layer_in = df.loc[model_name,'input'].astype('int')
    layer_1 = df.loc[model_name,'layer_1'].astype('int')
    layer_2 = df.loc[model_name,'layer_2'].astype('int')
    layer_3 = df.loc[model_name,'layer_3'].astype('int')
    learning_rate = df.loc[model_name,'learning_rate'].astype('float32')
    dropout_rate = df.loc[model_name,'dropout_rate'].astype('float32')
    return layer_in,layer_1,layer_2,layer_3,learning_rate,dropout_rate

def plot_results(model,model_var,model_type,date,df,sc):

    if(model_var == 'm'):
        #Multivariate
        in_size = 55
    else:
        #Univariate
        in_size = 45

    i1 = df.index.get_loc(date)
    plt.figure(figsize = (24,8))
    range_total = range(i1-5,i1+96*3)
    plt.subplots(5,1)

    for i in range(0,5):

        i1 += i*16 #Move forward by 4 hrs

        range_time = range(i1-15,i1)
        range_out = range(i1,i1+96)

        #Univariate
        input_recent = df['flow'].iloc[range_time]
        input_near = df['flow'].iloc[[r-96+int(15/2) for r in range_time]]
        input_distant = df['flow'].iloc[[r-96*4+int(15/2) for r in range_time]]  
        input_all = np.concat((input_recent,input_near,input_distant),axis=None)

        #Multivariate
        if(model_var=='m'):
            dow = df['day'].values.astype('int')
            holiday = df['holiday'].values.astype('int')
            time = df.index.strftime("%H%M").astype('int')
            input_dow = [0,0,0,0,0,0,0]
            input_dow[dow[i1]] = 1
            input_holiday = [holiday[i1],holiday[i1+96]]
            input_time = time[i1]/2400

            input_all = np.concat((input_all,input_dow,input_holiday,input_time),axis=None)

        in_size = input_all.shape[1]

        #Predict Flow
        if(model_type=='ANN'):
            pred_params = model.predict(sc.transform(input_all.values.reshape(-1,1)).reshape(1,in_size),verbose=0)
        else:
            pred_params = model.predict(sc.transform(input_all.values.reshape(-1,1)).reshape(1,in_size),verbose=0)
        
        flow_predict = norm.mean(pred_params[:,:96],pred_params[:,96:])
        flow_lower, flow_upper = norm.interval(0.9,pred_params[:,:96],pred_params[:,96:])
        
        #Destandardize
        flow_predict = sc.inverse_transform(flow_predict)
        flow_lower = sc.inverse_transform(flow_lower)
        flow_upper = sc.inverse_transform(flow_upper)

        flow_predict = np.concatenate((flow_predict,flow_lower,flow_upper),axis=0).transpose()
        flow_predict = pd.DataFrame(data=flow_predict,columns=['flow','lower','upper'],index=df.index[range_out])
        
        plt.subplot(5,1,i+1)
        plt.plot(df['flow'].iloc[range_total],label='Observed')
        plt.plot(df['flow'].iloc[range_time],label='Input')
        plt.plot(flow_predict['flow'],label='Predict')
        plt.fill_between(flow_predict.index,flow_predict['lower'],flow_predict['upper'],alpha=0.1,color='b',label = '90% Confidence')
        plt.xticks(fontsize = 6)
        plt.yticks(fontsize = 6)
        plt.legend(loc='upper right',fontsize="6")

    if(model_var=='m'):
        var_name = 'Multi'
    else:
        var_name = 'Uni'

    plt.suptitle(var_name + ' ' + model_type + ',' + date + ',' + pd.Timestamp(date).day_name())
    plt.tight_layout()

    return

def get_results(model_name):
    #Import data
    import initialize
    df = initialize.df

    #Convert data to numpy
    dataset = df['flow'].values
    dataset = dataset.astype('float32')

    #Split into train/test
    train_size = int(len(dataset)*0.66)
    test = dataset[train_size:]

    #Sequencing
    in_present, in_recent, in_distant = 15, 15, 15
    out_size = 96
    _, Y_test = split_sequence(test,in_present, in_recent, in_distant,out_size)

    X, Y = split_sequence(dataset,in_present, in_recent, in_distant,out_size)
    train_size = 11139
    X_train, Y_train = X[:train_size],Y[:train_size]
    X_test, Y_test = X[train_size:],Y[train_size:]

    output_time_arr = get_time_array(df)
    train_size = 11139
    output_time_arr = output_time_arr[train_size:]

    #Get model results
    results_path = r'C:\Users\darri\OneDrive - Imperial College London\Darrius_Cheong_TimeSesriesAnalysis\04_Code\Python\results'
    #model_name = 'm_ANNv4.xlsx'
    df_res = pd.read_excel((results_path+'\\'+model_name),sheet_name='Test')

    pred = np.array([np.array(i[1:-1].split(",")) for i in df_res['testPredict'].values]).astype('float')
    upper = np.array([np.array(i[1:-1].split(",")) for i in df_res['testupper'].values]).astype('float')
    lower = np.array([np.array(i[1:-1].split(",")) for i in df_res['testlower'].values]).astype('float')

    #Check scores
    testScore = scores(Y_test,pred,lower,upper)
    print(model_name)
    print(testScore.transpose())

    #Create Dataframe to store each predicted timestep
    #Create 2 time indexes
    #Time index 1 describes the first timestep in the forecast
    from itertools import repeat
    output_time_arr_1 = np.asarray([x for item in output_time_arr[:,0] for x in repeat(item, 96)])
    #Time index 2 describes the exact time being forecast
    output_time_arr_2=np.reshape(output_time_arr,(output_time_arr.shape[0]*output_time_arr.shape[1]))

    #Create dataframe describing results, first index refers to firsttimestep being forecast, second refers to individual timesteps
    testPredict_arr = np.reshape(pred,((pred.shape[0]*pred.shape[1])))
    lower_arr = np.reshape(lower,(lower.shape[0]*lower.shape[1]))
    upper_arr = np.reshape(upper,(upper.shape[0]*upper.shape[1]))
    Y_test_arr = np.reshape(Y_test,(Y_test.shape[0]*Y_test.shape[1]))
    df_mdl = pd.DataFrame(np.array([output_time_arr_1,output_time_arr_2,testPredict_arr,lower_arr,upper_arr,Y_test_arr,Y_test_arr-testPredict_arr]).transpose(),
                                columns=['initial_datetime','datetime','predict','lower','upper','actual','residuals'])
    df_mdl = df_mdl.astype({'predict':'float',
                            'lower':'float',
                            'upper':'float',
                            'actual':'float',
                            'residuals':'float',
                            })

    #Convert string to datetime
    df_mdl['initial_datetime'] = pd.to_datetime(df_mdl['initial_datetime'],dayfirst=True)
    df_mdl['datetime'] = pd.to_datetime(df_mdl['datetime'],dayfirst=True)
    df_mdl['datetime_date'] = [d.date() for d in df_mdl['datetime']]
    df_mdl['initial_date'] = [d.date() for d in df_mdl['initial_datetime']]

    #Result Analysis
    #1, Find most number of out of bounds by date
    df_mdl['under_bound'] = (df_mdl['actual']<df_mdl['lower'])
    df_mdl['over_bound'] = (df_mdl['actual']>df_mdl['upper'])
    df_mdl['out_bound'] = ((df_mdl['actual']>df_mdl['upper']) | (df_mdl['actual']<df_mdl['lower']))

    df_mdl['abs_residuals'] = df_mdl['residuals'].abs()

    return df_mdl

def get_results_arima(model_name):
    #Import data
    import initialize
    df = initialize.df

    #Convert data to numpy
    dataset = df['flow']

    #Split into train/test
    train_size = int(len(dataset)*0.66)
    Y_test = dataset[train_size:]
    Y_test_idx = Y_test.index.values
    Y_test = Y_test.values

    output_time_arr = get_time_array(df)
    output_time_arr = output_time_arr[train_size:]

    #Get model results
    results_path = r'C:\Users\darri\OneDrive - Imperial College London\Darrius_Cheong_TimeSesriesAnalysis\04_Code\Python\results'
    df_res = pd.read_excel((results_path+'\\'+model_name),sheet_name='Test')

    pred = df_res['testPredict'].values
    upper = df_res['testupper'].values
    lower = df_res['testlower'].values

    #Check scores
    testScore = scores(Y_test,pred,lower,upper)
    print(model_name)
    print(testScore.transpose())

    #Create Dataframe to store each predicted timestep
    #Create 2 time indexes
    #Create dataframe describing results, first index refers to firsttimestep being forecast, second refers to individual timesteps
    df_mdl = pd.DataFrame(np.array([Y_test_idx,Y_test_idx,pred,lower,upper,Y_test,Y_test-pred]).transpose(),
                                columns=['initial_datetime','datetime','predict','lower','upper','actual','residuals'])
    df_mdl = df_mdl.astype({'predict':'float',
                            'lower':'float',
                            'upper':'float',
                            'actual':'float',
                            'residuals':'float',
                            })

    #Convert string to datetime
    df_mdl['initial_datetime'] = pd.to_datetime(df_mdl['initial_datetime'],dayfirst=True)
    df_mdl['datetime'] = pd.to_datetime(df_mdl['datetime'],dayfirst=True)
    df_mdl['datetime_date'] = [d.date() for d in df_mdl['datetime']]
    df_mdl['initial_date'] = [d.date() for d in df_mdl['initial_datetime']]

    #Result Analysis
    #1, Find most number of out of bounds by date
    df_mdl['under_bound'] = (df_mdl['actual']<df_mdl['lower'])
    df_mdl['over_bound'] = (df_mdl['actual']>df_mdl['upper'])
    df_mdl['out_bound'] = ((df_mdl['actual']>df_mdl['upper']) | (df_mdl['actual']<df_mdl['lower']))

    df_mdl['abs_residuals'] = df_mdl['residuals'].abs()

    return df_mdl