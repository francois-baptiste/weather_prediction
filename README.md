# weather_prediction

This is the code I used for the 4th Module of the DSI (http://dsi-program.com/) program. 
It is a rainfall prediction code that uses conv-LSTM models to make prediction of rainfall in mm



The Radar data for this project was downloaded from the CIKM 2017 competition at the link below https://tianchi.aliyun.com/competition/information.htm?raceId=231596&_lang=en_US

The solution of this notebook is inspired by the solution given by:
https://github.com/TeaPearce/precipitation-prediction-convLSTM-keras which is itself an inspiration of the paper that uses convLSTM:
https://arxiv.org/abs/1506.04214

The data consisted of 1 training set, 2 testing sets which was 3.5GB in total when in compressed format. After uncompressing the data, the data which was stored in string format in text files came up to ~ 24GB of data. To work on the project though we need to feed the deep neural networks images - this requires to change the data to a HDF5 format which is essentially data cubes where each slice is an image. Transforming data to this format means that the images takes even more disk space. The training set would amount to > 50 GB of data with each testing set being ~ 10 GB of memory space. Below are a few functions that would transform the data from text format to the HDF5 format that is needed to train any tensorflow based model.
