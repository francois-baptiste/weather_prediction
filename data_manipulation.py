#!/usr/bin/env python
# coding: utf-8

# #### Data Manipulation

import datetime
from datetime import datetime
import h5py
import numpy as np


#### Function to just give some printing on the screen during conversion of file formats
def fn_print(string):
    print("\n-- ", string, ": ", datetime.now().strftime('%H:%M:%S'), "--")


def fn_load_data(path_in, no_import, start_line):
    id_no = []
    label = []
    data = []

    # scan through file line by line
    with open(path_in) as infile:
        i = -1
        for line in infile:
            i += 1
            if i % 500 == 0:
                fn_print(("considering line:" + str(i)))
            # print("considering line:",i)
            if i < start_line:
                continue
            if i >= no_import + start_line:
                break
            temp = line.split(",")
            id_no.append(str(temp[0]))
            label.append(float(temp[1]))
            data_temp = temp[2].split(" ")
            # data_temp = [int(x) for x in data_temp] # prob slowest part
            data_temp = list(map(int, data_temp))  # slightly quicker
            data.append(data_temp)

    # save results
    np_id = np.array(id_no)
    np_label = np.array(label)
    np_data = np.array(data)

    # clear memory
    del id_no, label, data

    # reshape data
    np_data = np.reshape(np_data, newshape=-1)
    T, H, Y, X = 15, 4, 101, 101
    np_data = np.reshape(np_data, newshape=(-1, T, H, Y, X), order='C')

    return np_id, np_label, np_data


## Function to append to a HDF5 file since keeping everything in memory would probably freeze computer.
def fn_h5_append(h5f_name, name_in, data_in):
    h5f = h5py.File(h5f_name, 'a')
    h5f.create_dataset(name_in, data=data_in)
    h5f.close()


# We can now convert all the files to HDF5 now by using the functions writen above.

########## converting testA to HDF5 using the functions given above
step_size = 2000
path_in = '/047efbea-741c-4d9b-90a7-128e39c9b91e1/data_new/testA.txt'
h5f_name = '/047efbea-741c-4d9b-90a7-128e39c9b91e1/data_new/data_testA.h5'
h5f = h5py.File(h5f_name, 'w')
grp = h5f.create_group('test_A')
h5f.close()

for i in np.arange(0, 2000, step_size):
    fn_print(("convert to h5, outter loop:" + str(i)))
    np_train_id, np_train_label, np_train_data = fn_load_data(path_in, start_line=i, no_import=step_size)
    filename_id = ('/test_A/test_id_' + str(i) + "_to_" + str(i + step_size - 1))
    filename_label = ('/test_A/test_label_' + str(i) + "_to_" + str(i + step_size - 1))
    filename_data = ('/test_A/test_data_' + str(i) + "_to_" + str(i + step_size - 1))
    ascii_id = [n.encode("ascii", "ignore") for n in np_train_id.tolist()]

    fn_h5_append(h5f_name, filename_id, ascii_id)
    fn_h5_append(h5f_name, filename_label, np_train_label)
    fn_h5_append(h5f_name, filename_data, np_train_data)

    del np_train_id, np_train_label, np_train_data

########## converting testB to HDF5
path_in = '/home/ubuntu/data_new/CIKM2017_testB/testB.txt'
h5f_name = '/home/ubuntu/data_new/CIKM2017_testB/data_testB.h5'
h5f = h5py.File(h5f_name, 'w')
grp = h5f.create_group('test_B')
h5f.close()

for i in np.arange(0, 2000, step_size):
    fn_print(("convert to h5, outter loop:" + str(i)))
    np_train_id, np_train_label, np_train_data = fn_load_data(path_in, start_line=i, no_import=step_size)
    filename_id = ('/test_B/test_id_' + str(i) + "_to_" + str(i + step_size - 1))
    filename_label = ('/test_B/test_label_' + str(i) + "_to_" + str(i + step_size - 1))
    filename_data = ('/test_B/test_data_' + str(i) + "_to_" + str(i + step_size - 1))
    ascii_id = [n.encode("ascii", "ignore") for n in np_train_id.tolist()]

    fn_h5_append(h5f_name, filename_id, ascii_id)
    fn_h5_append(h5f_name, filename_label, np_train_label)
    fn_h5_append(h5f_name, filename_data, np_train_data)

    del np_train_id, np_train_label, np_train_data

########## converting train set to HDF5 using only 4,000 samples instead of the total 10,000
step_size = 4000
path_in = '/home/ubuntu/data_new/CIKM2017_train/train.txt'
h5f_name = '/home/ubuntu/data_new/CIKM2017_train/train.h5'
h5f = h5py.File(h5f_name, 'w')
grp = h5f.create_group('train')
h5f.close()

for i in np.arange(0, 4000, step_size):
    fn_print(("convert to h5, outter loop:" + str(i)))
    np_train_id, np_train_label, np_train_data = fn_load_data(path_in, start_line=i, no_import=step_size)
    filename_id = ('/train/train_id_' + str(i) + "_to_" + str(i + step_size - 1))
    filename_label = ('/train/train_label_' + str(i) + "_to_" + str(i + step_size - 1))
    filename_data = ('/train/train_data_' + str(i) + "_to_" + str(i + step_size - 1))
    ascii_id = [n.encode("ascii", "ignore") for n in np_train_id.tolist()]

    fn_h5_append(h5f_name, filename_id, ascii_id)
    fn_h5_append(h5f_name, filename_label, np_train_label)
    fn_h5_append(h5f_name, filename_data, np_train_data)

    del np_train_id, np_train_label, np_train_data

############# creating a validation set from the training set
#### using a portion of data that does not overlap with the reduced test set. taking 500 sequencial samples
#### as the validation set.

path_in = '/home/ubuntu/data_new/CIKM2017_train/train.txt'
h5f_name = '/home/ubuntu/data_new/CIKM2017_train/val.h5'
h5f = h5py.File(h5f_name, 'w')
grp = h5f.create_group('val')
h5f.close()
step_size = 500
for i in np.arange(5000, 5500, step_size):
    fn_print(("convert to h5, outter loop:" + str(i)))
    np_train_id, np_train_label, np_train_data = fn_load_data(path_in, start_line=i, no_import=step_size)
    filename_id = ('/train/val_id_' + str(i) + "_to_" + str(i + step_size - 1))
    filename_label = ('/train/val_label_' + str(i) + "_to_" + str(i + step_size - 1))
    filename_data = ('/train/val_data_' + str(i) + "_to_" + str(i + step_size - 1))
    ascii_id = [n.encode("ascii", "ignore") for n in np_train_id.tolist()]

    fn_h5_append(h5f_name, filename_id, ascii_id)
    fn_h5_append(h5f_name, filename_label, np_train_label)
    fn_h5_append(h5f_name, filename_data, np_train_data)

    del np_train_id, np_train_label, np_train_data

# We now have the data in the format we wanted, we can focus on some data preparation and the modelling part of the work.
