#!/usr/bin/env python
# coding: utf-8

from radar_model_tools import fn_Xy_to_tframe, fn_get_model_convLSTM_tframe_5
from keras import backend as K
from sklearn.utils import shuffle
from tools import fn_h5_to_Xy_2D_timeD

def fn_keras_rmse(y_true, y_pred, y_std=15., ):
    return K.sqrt(K.mean(K.square((y_pred * y_std) - (y_true * y_std))))

# By using this method instead of having 4000 sequences (from the 4000 training samples) we get 12000 image sequences.
# To properly randomise the input to the algorithm, sklearn shuffle is used.

# The following code was run to train the algorithm at all the heights of the radar images


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

    print('Training model')
    is_graph=False


    history = model.fit(X0_train, X1_train[:, 0, :, :, :], batch_size=10,
                        epochs=3, verbose=1, validation_data=(X0_t_val, X1_t_val[:, 0, :, :, :]))


    model.save_weights("model_convLSTM_tframe5_train4000_val500_height " + str(i) + "gauss_1_mse.h5")


    # import matplotlib.pyplot as plt
    # fig, ax1 = plt.subplots(1, 1)
    # ax1.plot(history.history["val_loss"])
    # ax1.plot(history.history["loss"])



