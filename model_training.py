#!/usr/bin/env python
# coding: utf-8

# #### Data Manipulation

from tools import fn_h5_to_Xy_2D_timeD, fn_run_model, fn_get_model_convLSTM_2

# We now have all the tools to do the modelling. Here the work is splitted into 2 sections. First we want to predict the amount of rainfall at the desired location (centre of images) when given a sequence of radar images. The other, part of the work which is the visualisation of the radar images i.e predicting the next radar image given a sequence of images. This is carries less weight in the prediction of rainfall but gives a direct visual feedback on how good this model works.

# setting a few global parameters that are used to scale the data
y_std = 15.
y_mean = 15.
X_std = 50.
X_mean = 60.
my_height = 3

####### RUNNING THE MODEL

print('loading data')
X_train, y_train = fn_h5_to_Xy_2D_timeD(test_train="train", i=0, h_select=my_height)
X_t_val, y_t_val = fn_h5_to_Xy_2D_timeD(test_train="val", i=4, h_select=my_height)

print('loading model')
model = fn_get_model_convLSTM_2()

print(model.summary())
model.compile(loss='mean_squared_error', optimizer='adam')

print('Training model')
fn_run_model(model, X_train, y_train, X_t_val, y_t_val, batch_size=8, nb_epoch=15, verbose=1, is_graph=True)

print('saving model')
model.save_weights('model_convLSTM_2_train4000_val500.h5')