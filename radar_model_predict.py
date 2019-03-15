#!/usr/bin/env python
# coding: utf-8

import numpy as np
from keras import backend as K
from sklearn.utils import shuffle

from radar_model_tools import fn_Xy_to_tframe, fn_get_model_convLSTM_tframe_5
from tools import fn_h5_to_Xy_2D_timeD


def fn_keras_rmse(y_true, y_pred, y_std=15., ):
    return K.sqrt(K.mean(K.square((y_pred * y_std) - (y_true * y_std))))


# From the results we can create a little animation of the predicted motion of clouds. This is done using the code below

for i in range(0, 4):
    print('loading data')
    my_height = i
    X_train, y_train = fn_h5_to_Xy_2D_timeD(test_train="train", i=0, h_select=my_height)
    X_t_val, y_t_val = fn_h5_to_Xy_2D_timeD(test_train="val", i=4, h_select=my_height)

    X0_train, X1_train = fn_Xy_to_tframe(X_train)
    X0_t_val, X1_t_val = fn_Xy_to_tframe(X_t_val)

    # X0_train, X1_train = fn_Xy_to_tframe_v2(X_train)
    # X0_t_val, X1_t_val = fn_Xy_to_tframe_v2(X_t_val)

    print('Shuffling')
    X0_train, X1_train = shuffle(X0_train, X1_train, random_state=0)
    X0_t_val, X1_t_val = shuffle(X0_t_val, X1_t_val, random_state=0)

    print('loading model')
    model = fn_get_model_convLSTM_tframe_5()

    print(model.summary())
    # model.compile(loss='binary_crossentropy', optimizer='adam')
    # model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss=fn_keras_rmse, optimizer='adam')

    model.load_weights("model_convLSTM_tframe5_train4000_val500_height " + str(i) + "gauss_1_mse.h5")

    input_frames = 5
    s_select = 0
    output_frames = 10
    X_input = X0_train[s_select, :input_frames, :, :, :]
    X_input1 = np.zeros((X_input.shape[0] + output_frames, X_input.shape[1], X_input.shape[2], X_input.shape[3]), float)
    X_input1[0:5] = X_input
    for i in range(0, output_frames):
        print(i)
        X_pred = model.predict(X_input1[i: i + 5, :, :, :].reshape(1, 5, 101, 101, 1))  # predict
        X_input1[5 + i] = X_pred[0, :, :, :]

    np.save('X_input1' + str(i) + '.npy', X_input1)
