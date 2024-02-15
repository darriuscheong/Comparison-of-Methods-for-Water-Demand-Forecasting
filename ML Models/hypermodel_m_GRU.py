# import required packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, BatchNormalization
from tensorflow.keras.layers import ReLU, MaxPool2D, AvgPool2D, GlobalAvgPool2D
from tensorflow.keras.utils import plot_model
import keras_tuner as kt
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from scipy.stats import nbinom, norm
import time as time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Lambda, LSTM, Dropout, GRU, Concatenate
from tensorflow.keras.models import Model, Sequential

#import sys
#sys.path.append(r'C:\Users\darri\OneDrive - Imperial College London\Darrius_Cheong_TimeSesriesAnalysis\04_Code\Python')

#Import functions
import glob_func
from glob_func import scores, split_sequence,split_sequence_exog, split_exog

#Import Distribution functions
import dist_func
from dist_func import gaussian_distribution_layer, gaussian_loss

#Import data
import initialize
df = initialize.df

#Convert data to numpy
dataset = df['flow'].values
dataset = dataset.astype('float32')

#Split into train/test
train_size = int(len(dataset)*0.66)
test_size = len(dataset)-train_size

train, test = dataset[:train_size],dataset[train_size:]

#Standardize
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_norm = sc.fit_transform(train.reshape(-1,1))
test_norm = sc.transform(test.reshape(-1,1))

df_train_norm = df[:train_size]
df_train_norm.loc[:,'flow'] = train_norm
df_test_norm = df[train_size:]
df_test_norm.loc[:,'flow'] = test_norm

#Sequencing
in_present, in_recent, in_distant = 15, 15, 15
in_size = in_present + in_recent + in_distant
out_size = 96
X_train, Y_train = split_sequence_exog(df_train_norm,in_present, in_recent, in_distant,out_size)
X_test, Y_test = split_sequence_exog(df_test_norm,in_present, in_recent, in_distant,out_size)

#Exogenous
X_train_exog = split_exog(df[:train_size])
X_test_exog = split_exog(df[train_size:])

#For LSTM, change shape
n_features = 1
X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],n_features))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],n_features))

#Define hypermodel
def build_model(hp):
    dropout = hp.Float('dropout_rate',min_value=0.1,max_value=0.5,step=0.05,default=0.2)

    #Time Inputs
    time_input = Input(shape=(X_train.shape[1], 1),name="time")
    
    time_features = GRU(hp.Int('nodes_input',min_value = 32,max_value = 256, step=32), activation = 'relu', return_sequences=True)(time_input)
    time_features = GRU(hp.Int('nodes_1',min_value = 32,max_value = 256, step=32), activation = 'relu',dropout=dropout)(time_features)

    #Exog Inputs
    exog_input = Input(shape=(X_train_exog.shape[1], ),name="exog")
    
    #Merge exog and x
    x = Concatenate(axis=1)([time_features,exog_input])

    #Dense layers for more feature extraction
    x = Dense(hp.Int('nodes_2',min_value = 32,max_value = 256, step=32), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(hp.Int('nodes_3',min_value = 32,max_value = 256, step=32), activation='relu')(x)

    #Gaussian Distribution
    outputs = Dense(out_size*2)(x)
    distribution_outputs = Lambda(gaussian_distribution_layer)(outputs)
    
    #Define model
    model = Model(inputs=[time_input,exog_input], outputs=distribution_outputs)

    #Tune learning rate from 0.0001 to 0.01
    #opt = Adam(hp.Float('learning_rate',min_value=1e-4,max_value=1e-2,sampling='log'))
    opt = Adam(tf.keras.optimizers.schedules.InverseTimeDecay(
                    hp.Float('learning_rate',min_value=1e-4,max_value=1e-2,sampling='log'),
                    decay_steps=train_size*1000,
                    decay_rate=1,
                    staircase=False),
                    clipvalue=0.5,
                    clipnorm=1
                    )
    
    model.compile(loss = gaussian_loss, optimizer = opt)

    return model

#Initialize Hyperband search algo tuner
tuner = kt.Hyperband(
    hypermodel=build_model,
    objective='val_loss',
    max_epochs=30,
    hyperband_iterations=2,
    directory="tuner_dir",
    project_name="m_GRUv4",
    max_consecutive_failed_trials=20
)

#Run search
'''
tuner.search([X_train,X_train_exog],Y_train,validation_data=([X_test,X_test_exog],Y_test),
             epochs=30,
             verbose=2)
'''
#tuner.results_summary()

# Get the optimal hyperparameters
best_hps= tuner.get_best_hyperparameters(1)[0]

# get the best model
best_model = tuner.get_best_models(1)[0]

# show model summary
print('Optimized Multi GRUv4')
best_model.summary()

#Get opt learning rate
from keras import backend as K
learning_rate = K.eval(best_model.optimizer.lr)
dropout_rate = best_model.get_layer(index=2).get_config()['dropout']
print('Learning rate = %f' % learning_rate)
print('Dropout = %f' % dropout_rate)

#Train Based on optimal model
model = tuner.hypermodel.build(best_hps)

#Train Model
checkpoint_path = './checkpoints/m_GRUv4_200'
mdl_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    save_weights_only=True,
    monitor='val_loss',
    mode ='min',
    save_best_only=True
)

start=time.time()
model.fit([X_train,X_train_exog], Y_train, epochs = 200,
                    validation_data = ([X_test,X_test_exog],Y_test),
                    verbose=2,
                    #callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=201,restore_best_weights=True)]
                    callbacks = [mdl_checkpoint_cb] 
                    )
train_time = (time.time()-start)

model.load_weights(checkpoint_path)

#Save model weights
#model.save_weights('m_GRU_weights')

#Evaluate Test Data
pred_params = model.predict([X_test,X_test_exog],verbose=0)
testPredict = norm.mean(pred_params[:,:96],pred_params[:,96:])
lower, upper = norm.interval(0.9,pred_params[:,:96],pred_params[:,96:])

#Evaluate Train Data
pred_params = model.predict([X_train,X_train_exog],verbose=0)
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

score_arr = [trainScore['value'][0], trainScore['value'][1],trainScore['value'][2],trainScore['value'][3],trainScore['value'][4],
                 testScore['value'][0], testScore['value'][1],testScore['value'][2],testScore['value'][3],testScore['value'][4]]

print('m_GRUv4',end=',')
for i in score_arr:
    print(i,end=',')
print(train_time)

