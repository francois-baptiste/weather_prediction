#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from rainfall_model_tools import fn_get_model_convLSTM_2
from keras import backend as K
from tools import fn_h5_to_Xy_2D_timeD

def fn_keras_rmse(y_true, y_pred, y_std=15., ):
    return K.sqrt(K.mean(K.square((y_pred * y_std) - (y_true * y_std))))

sess = tf.Session()
K.set_session(sess)

# setting a few global parameters that are used to scale the data
y_std = 15.
y_mean = 15.
X_std = 50.
X_mean = 60.
my_height = 3

# #### Evaluating the model
#
# To evaluate the model we created, we load back the weights of the model and then send the test data to the model and get rainfall prediction. From there we use the same RMSE metric as was used in the CIKM 2017 competition to evaluate the accuracy of the model created


print('loading model')
model = fn_get_model_convLSTM_2()

### loading weights
model.load_weights('model_convLSTM_2_train4000_val500.h5')

### Loop to evaluate the model at each height for test set A
testA = []
with sess.as_default():
    for i in range(4):
        height_rem = [0, 1, 2, 3]
        print(i)

        t = height_rem[i]
        X_test, y_test = fn_h5_to_Xy_2D_timeD(test_train="testA", i=0, h_select=my_height)
        y_pred = model.predict(X_test)
        testA.append(fn_keras_rmse(y_test, y_pred).eval())
    print(testA)


# testA = [13.5037568326104, 13.569423492251511, 13.654292392613222, 13.208109666576338]
#
# mean RMSE for test A = 13.48389
#
# testB = [13.544123493977148, 13.478812867719338, 13.37080457808451, 13.08367986944559]
#
# mean RMSE for test B = 13.36935
#
# Overall RMSE for all the different all test data sets at all height is
#
# Overall RMSE = 13.43 (done by averaging all the 8 different RMSE)
#
# This score would be ranked in the top 20 for the CIKM 2017 competition!

# #### Predicting the next Radar image
#
# As an aside to this project, one could use the same principle of convLSTM deep NN model to predict the next radar images after feeding the algorithm a sequence of images. This follows the same principle as the moving digit of MNIST challenge (http://www.cs.toronto.edu/~nitish/unsupervised_video/)
#
# Below is the model that we settle to use for this section of the project
#


