import h5py
import matplotlib.pyplot as plt

# from h5 file to Xy - 2D + timeD, at one height
# function to load images both for X and Y - used for next radar image prediction

y_std = 15.
y_mean = 15.
X_std = 50.
X_mean = 60.
my_height = 3


#### Data loader function - use to select which HDF5 file to read and load
def fn_h5_to_np(test_train, i=0):
    if test_train == "testA":
        filename_id = ('/test_A/test_id_0_to_1999')
        filename_label = ('/test_A/test_label_0_to_1999')
        filename_data = ('/test_A/test_data_0_to_1999')
        h5f_name = '/home/ubuntu/data_new/CIKM2017_testA/data_testA.h5'

    elif test_train == "testB":
        filename_id = ('/test_B/test_id_0_to_1999')
        filename_label = ('/test_B/test_label_0_to_1999')
        filename_data = ('/test_B/test_data_0_to_1999')
        h5f_name = '/home/ubuntu/data_new/CIKM2017_testB/data_testB.h5'

    elif test_train == "train":
        filename_id = ('/train/train_id_0_to_3999')
        filename_label = ('/train/train_label_0_to_3999')
        filename_data = ('/train/train_data_0_to_3999')
        h5f_name = '/home/ubuntu/data_new/CIKM2017_train/train.h5'

    elif test_train == "val":
        filename_id = ('/train/val_id_5000_to_5499')
        filename_label = ('/train/val_label_5000_to_5499')
        filename_data = ('/train/val_data_5000_to_5499')
        h5f_name = '/home/ubuntu/data_new/CIKM2017_train/val.h5'

    h5f = h5py.File(h5f_name, 'r')
    np_id = h5f[filename_id][:]
    np_label = h5f[filename_label][:]
    np_data = h5f[filename_data][:]
    h5f.close()

    return np_id, np_label, np_data


# normalise input and output
def fn_norm_Xy(X, y, X_std = 50., y_std = 15., is_graph=False):
    # normalise X and y values
    X = X / X_std
    y = y / y_std
    if is_graph:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.hist(X.reshape(-1), bins=100)
        ax2.hist(y.reshape(-1), bins=100)
    return X, y


# remove negative values in images - clipping every negative value to be zero
def fn_minus_ones(X):
    X[(X < 0)] = 0
    return X


# convert np arrays to X and y
def fn_np_to_Xy(np_data, np_label, h_select=3, t_select=14):
    X = np_data[:, t_select, h_select, :, :]  # select all samples, all data
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    y = np_label
    y = y.reshape(y.shape[0], 1)

    return X, y


# from h5 file to Xy 2D, at one height - This function is use to load data that matches radar image with
## recorded amount of rainfall
def fn_h5_to_Xy(test_train, i=0, h_select=3, t_select=14):
    ###### using data loader function to load data
    np_id, np_label, np_data = fn_h5_to_np(test_train=test_train, i=i)
    X, y = fn_np_to_Xy(np_data, np_label, h_select=h_select, t_select=t_select)
    X = fn_minus_ones(X)
    X, y = fn_norm_Xy(X, y, is_graph=False)  # normalise
    return X, y


# convert np arrays to X and y - 2D + timeD, at one height
def fn_np_to_Xy_2D_timeD(np_data, np_label, h_select=3):
    X = np_data[:, :, h_select, :, :]  # select all samples, all data
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1)
    y = np_label
    y = y.reshape(y.shape[0], 1)

    return X, y


# from h5 file to Xy - 2D + timeD, at one height
# function to load images both for X and Y - used for next radar image prediction
def fn_h5_to_Xy_2D_timeD(test_train, i=0, h_select=3):
    np_id, np_label, np_data = fn_h5_to_np(test_train=test_train, i=i)  # load data from h5
    X, y = fn_np_to_Xy_2D_timeD(np_data, np_label, h_select=h_select)  # convert to X,y format
    X = fn_minus_ones(X)
    X, y = fn_norm_Xy(X, y, is_graph=False)  # normalise
    return X, y


def fn_h5_to_Xy_2D_timeD(test_train, i=0, h_select=3):
    np_id, np_label, np_data = fn_h5_to_np(test_train=test_train, i=i)  # load data from h5
    X, y = fn_np_to_Xy_2D_timeD(np_data, np_label, h_select=h_select)  # convert to X,y format
    X = fn_minus_ones(X)
    X, y = fn_norm_Xy(X, y, is_graph=False)  # normalise
    return X, y


#### loading the keras libraries
from keras.layers import Dense, MaxPooling2D, Dropout, BatchNormalization, Flatten, Conv2D
from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional_recurrent import ConvLSTM2D


def fn_keras_rmse(y_true, y_pred, y_std = 15., ):
    return K.sqrt(K.mean(K.square((y_pred * y_std) - (y_true * y_std))))


def fn_run_model(model, X, y, X_val, y_val, batch_size=50, nb_epoch=40, verbose=2, is_graph=False):
    history = model.fit(X, y, batch_size=batch_size,
                        epochs=nb_epoch, verbose=verbose, validation_data=(X_val, y_val))
    if is_graph:
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(history.history["val_loss"])
        ax1.plot(history.history["loss"])


#  convLSTM model to predict the LSTM network
def fn_get_model_convLSTM_2():
    model = Sequential()

    model.add(ConvLSTM2D(filters=32, kernel_size=(7, 7),
                         input_shape=(None, 101, 101, 1),
                         return_sequences=True,
                         go_backwards=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.2
                         ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=16, kernel_size=(7, 7),
                         return_sequences=True,
                         go_backwards=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.4, recurrent_dropout=0.2
                         ))
    model.add(BatchNormalization())

    model.add(ConvLSTM2D(filters=8, kernel_size=(7, 7),
                         return_sequences=False,
                         go_backwards=True,
                         activation='tanh', recurrent_activation='hard_sigmoid',
                         kernel_initializer='glorot_uniform', unit_forget_bias=True,
                         dropout=0.3, recurrent_dropout=0.2
                         ))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=1, kernel_size=(1, 1),
                     activation='relu',
                     data_format='channels_last'))

    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(1, activation='linear'))

    print(model.summary())

    return model
