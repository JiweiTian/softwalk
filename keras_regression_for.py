#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 13:27:02 2019

@author: wutong
"""

from utilize import loaddata, error_func, loss_func
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
import time
import numpy as np
from keras.layers import Dense, Activation, Flatten
import scipy.io as sio

X_name='./data_journal/case2000_mc_X_trun_48_norm_wt35.mat'
Y_name='./data_journal/MC_real.mat'
V_name='VA_row'


split_ratio=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
for i in split_ratio:
	X_train, X_test, y_train, y_test, Y_std, Y_mean, perm_train, perm_test=loaddata(X_name, Y_name, i, V_name)
#%%##########

	NN_model = Sequential()

# The Input Layer :
	NN_model.add(Dense(256, kernel_initializer='normal',input_dim = X_train.shape[1], activation='sigmoid'))

# The Hidden Layers :
	NN_model.add(Dense(256, kernel_initializer='normal',activation='sigmoid'))
	NN_model.add(Dense(256, kernel_initializer='normal',activation='sigmoid'))
	NN_model.add(Dense(256, kernel_initializer='normal',activation='sigmoid'))

# The Output Layer :
	NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile:
	NN_model.compile(loss=loss_func, optimizer='adam', metrics=['mean_absolute_error'])


	print('%%%%%%%%%%%', i, '%%%%%%%%%%%')
	NN_model.fit(X_train, y_train, epochs=800, batch_size=100, validation_split = 0.1, verbose=0, use_multiprocessing=True)
	y_pred = NN_model.predict(X_test)
	V_test=(y_test*Y_std+Y_mean)
	V_pred=(y_pred*Y_std+Y_mean)
	V_train=(y_train*Y_std+Y_mean)
	score=error_func(V_test, V_pred)
	print('Score:', score)
	savename='./data/VA_model1_'+str(int(i*100))+'.mat'
	sio.savemat(savename, {'V_train':V_train, 'V_test': V_test, 'V_pred': V_pred, 'perm_train': perm_train, 'perm_test':perm_test})
	print('Save finished!')