#!/usr/bin/env python
# coding: utf-8

from keras.layers import BatchNormalization, Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.models import Sequential
from scipy.ndimage import filters
import numpy as np

def fn_get_model_convLSTM_tframe_5():
    model = Sequential()
    model.add(ConvLSTM2D(filters=64, kernel_size=(7, 7),
                         input_shape=(None, 101, 101, 1), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.3, recurrent_dropout=0.3, go_backwards=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7), padding='same', return_sequences=False,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.3, go_backwards=True))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                     activation='sigmoid',
                     padding='same', data_format='channels_last'))

    print(model.summary())
    return model


# The idea here is to feed 5 images to the network and the network is suppose to predict the 6th image of the sequence. Hence the Y input to this part need to be the 6th image in any sequence. Therefore another function needs to be written that would take the training set and give a sequence of 5 images to X_input and 1 image to Y_input. The following function does that:


def fn_Xy_to_tframe(X_in):
    n_frames = 5
    n_frames_out = 1
    X0 = np.zeros(X_in.shape, dtype=np.float)
    X0 = X0[:, 0:n_frames, :, :, :]  # only use 5 frames each
    X1 = np.zeros(X_in.shape, dtype=np.float)
    X1 = X1[:, 0:n_frames_out, :, :, :]

    for sample in range(0, X_in.shape[0]):

        X0[sample] = X_in[sample, 0:n_frames, :, :, :]
        X1[sample] = X_in[sample, n_frames:n_frames + n_frames_out,:,:,:]

        # blur inputs
        for i in range(n_frames):
            X0[sample, i, :, :, 0] = filters.gaussian_filter(X0[sample, i, :, :, 0], sigma=1.8)
        for i in range(n_frames_out):
            X1[sample, i, :, :, 0] = filters.gaussian_filter(X1[sample, i, :, :, 0], sigma=1.8)

    X_max = 3.0
    X0 = np.clip(X0, 0, X_max)  # clip
    X1 = np.clip(X1, 0, X_max)
    X0 = X0 / X_max
    X1 = X1 / X_max

    return X0, X1


# The only problem with this function is that it only takes the first 5 images of every sequence of the training set. Each of the training set sequence are 15 images long. We therefore rewrite the same function to pick the start 5+1 (X=5, Y=1) image sequence, the middle 5+1 sequence and the last 5+1 image sequence. In this way we sample more or less the entire sequences without losing data - allowing the algorithm to generalise in principle. The following function is an extension of the above one


def fn_Xy_to_tframe_v2(X_in):
    n_frames = 5  # X frames
    n_frames_out = 1  # Y frames

    X0_0 = np.zeros(X_in.shape, dtype=np.float)
    X0_0 = X0_0[:, 0:n_frames, :, :, :]
    X0_1 = np.copy(X0_0)
    X0_2 = np.copy(X0_0)

    X1_0 = np.zeros(X_in.shape, dtype=np.float)
    X1_0 = X1_0[:, 0:n_frames_out, :, :, :]
    X1_1 = np.copy(X1_0)
    X1_2 = np.copy(X1_0)

    for sample in range(0, X_in.shape[0]):

        X0_0[sample] = X_in[sample, 0:n_frames, :, :, :]
        X0_1[sample] = X_in[sample, 5: 5 + n_frames, :, :, :]
        X0_2[sample] = X_in[sample, 9: 9 + n_frames, :, :, :]

        X1_0[sample] = X_in[sample, n_frames:n_frames + n_frames_out,:,:,:]
        X1_1[sample] = X_in[sample, 5 + n_frames: 5 + n_frames + n_frames_out,:,:,:]
        X1_2[sample] = X_in[sample, 9 + n_frames: 9 + n_frames + n_frames_out,:,:,:]


        for i in range(n_frames):
            X0_0[sample, i, :, :, 0] = filters.gaussian_filter(X0_0[sample, i, :, :, 0], sigma=1.0)
            X0_1[sample, i, :, :, 0] = filters.gaussian_filter(X0_1[sample, i, :, :, 0], sigma=1.0)
            X0_2[sample, i, :, :, 0] = filters.gaussian_filter(X0_2[sample, i, :, :, 0], sigma=1.0)
        for i in range(n_frames_out):
            X1_0[sample, i, :, :, 0] = filters.gaussian_filter(X1_0[sample, i, :, :, 0], sigma=1.0)
            X1_1[sample, i, :, :, 0] = filters.gaussian_filter(X1_1[sample, i, :, :, 0], sigma=1.0)
            X1_2[sample, i, :, :, 0] = filters.gaussian_filter(X1_2[sample, i, :, :, 0], sigma=1.0)



    X_max = 3.0
    X0_0 = np.clip(X0_0, 0, X_max)  # clip
    X0_1 = np.clip(X0_1, 0, X_max)
    X0_2 = np.clip(X0_2, 0, X_max)

    X1_0 = np.clip(X1_0, 0, X_max)
    X1_1 = np.clip(X1_1, 0, X_max)
    X1_2 = np.clip(X1_2, 0, X_max)

    X0_0 = X0_0 / X_max
    X0_1 = X0_1 / X_max
    X0_2 = X0_2 / X_max

    X1_0 = X1_0 / X_max
    X1_1 = X1_1 / X_max
    X1_2 = X1_2 / X_max

    ### combining all 3 arrays together
    X0 = np.copy(X0_0)
    X0 = np.concatenate((X0, X0_1, X0_2), axis=1)

    X1 = np.copy(X1_0)
    X1 = np.concatenate((X1, X1_1, X1_2), axis=1)
    return X0, X1