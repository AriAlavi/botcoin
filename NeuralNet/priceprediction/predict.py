import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Normalization
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

import time
import datetime
#This is for our datasets for keras/tensorflow, here I use it to create our price prediction dataset.
#One col is for the future price, rest are inputs such as current price, mean, etc
import pandas as pd

#this is actually mostly unused, ignore comments
#since short is hourly, 24 rows would be 1 day in the future. 7 rows for long is 1 week
def cleanframe(filename, rows):
    dataframe = pd.read_excel(filename)
    dataframe.iloc[:,0] = dataframe['price'].shift(-rows)
    dataframe = dataframe.rename(columns = {'Unnamed: 0' : 'futurePrice'})
    #remove last n rows because they have nofuture price
    dataframe.drop(dataframe.tail(rows).index,inplace=True)
    dataframe.drop(dataframe.head(rows).index,inplace=True)
    dataframe.drop('endDate', axis = 1, inplace=True)
    dataframe.drop('date', axis = 1, inplace=True)
    return dataframe
    

dataframe = cleanframe("shaved.xlsx", 7)
dataframe.drop(['futurePrice'],axis=1, inplace=True)


sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(dataframe)


import joblib
scaler_filename = "trainscaler.save"
joblib.dump(sc, scaler_filename) 

#split 80-20 training vs testing
train = np.array(training_set_scaled[:int(np.round(len(training_set_scaled)*.8))])
test = np.array(training_set_scaled[int(np.round(len(training_set_scaled)*.8)):])



#shapes the dataset for training the model, it's a look_back approach
#model gets fed in batches of 7 days of data
def create_dataset(dataset, look_back):
    x,y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),:]
        x.append(a)
        y.append(dataset[i+look_back,5])
    return np.array(x), np.array(y)


xtrain,ytrain = create_dataset(train, 7)
xtest,ytest = create_dataset(test,  7)


xtrain = np.reshape(xtrain, (xtrain.shape))
xtest = np.reshape(xtest, (xtest.shape))

name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "shaved"


log_dir = "logs/fit/" + name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)


# model = Sequential()
# model.add(LSTM(30, input_shape=(xtrain.shape[1],xtrain.shape[2]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(30, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(units=30))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# history = model.fit(xtrain, ytrain, validation_data=[xtest,ytest], callbacks=[tensorboard_callback], epochs=400, batch_size=64)
# model.save('400epoch64b' + name)








#this line above is where the model is trained and saved

#load the model

# model = keras.models.load_model('400epoch64b' + name)
model = keras.models.load_model('400epoch64b20220214-143742shaved')

#predicting prices and stuffing them back into an array which can be scale inverted, total pain and complicated but required due to how normalization works
#I made a function to make this easier but its in another file

traindf_min = sc.data_min_
traindf_max = sc.data_min_

def unscale(value, scaler):
    traindf_min = scaler.data_min_[0]
    traindf_max = scaler.data_max_[0]
    return ((value * (traindf_max - traindf_min)) + traindf_min)

def scale(value, scaler):
    traindf_min = scaler.data_min_[0]
    traindf_max = scaler.data_max_[0]
    return (value - traindf_min) / (traindf_max-traindf_min)



traindf_min = sc.data_min_
traindf_max = sc.data_min_

#5 is the index of the price volumn
def unscale(value, scaler):
    traindf_min = scaler.data_min_[5]
    traindf_max = scaler.data_max_[5]
    return ((value * (traindf_max - traindf_min)) + traindf_min)

def scale(value, scaler):
    traindf_min = scaler.data_min_[5]
    traindf_max = scaler.data_max_[5]
    return (value - traindf_min) / (traindf_max-traindf_min)


testmodel = model.predict(xtest)
testvalues = unscale(testmodel,sc)

trainmodel = model.predict(xtrain)
trainvalues = unscale(trainmodel,sc)


wholetest,wholetesty = create_dataset(training_set_scaled, 7)
wholetest = np.reshape(wholetest, (wholetest.shape))
wholemodel = model.predict(wholetest)
wholevalues = unscale(wholemodel, sc)




#blue is predicted
plt.plot(range(len(wholevalues)), wholevalues)
 #orange is 
plt.plot(range(len(dataframe['price'])), dataframe['price'])

# plt.plot(range(0,len(testvalues)), tespltvalues)

plt.show()

from sklearn.externals import joblib
scaler_filename = "trainscaler.save"
joblib.dump(sc, scaler_filename) 



sc = joblib.load(scaler_filename) 

#test inputs for saved model

def scaleinputsandpredict(inputs):
    scaledf = sc.transform(df)
    scaledx = np.array(scaledf[0:7,:])
    scaledx = np.reshape(scaledx, (1, scaledx.shape[0], scaledx.shape[1]))
    predictedscaled = model.predict(scaledx)
    return unscale(predictedscaled, sc)


volume = [29237.52176779,24822.90047457,38575.1201996,91246.5487664299,27477.59553625,12654.57691273,5678.91496107001]
low = [9.8461,10.10007,10.20136,9.16101,9.19291,9.56651,9.60162]
high = [10.47,10.84499,10.7069,10.62,10.18991,9.83,9.89999]
open =[9.9134,10.35,10.3311,10.6098,9.25511,9.60019,9.6921]
close = [10.354,10.31113,10.6195,9.16101,9.6,9.6965,9.8327]
price = [10.145875,10.4015475,10.464715,9.887955,9.5594825,9.6733,9.7566025]
transactions = [661,889,661,2296,688,364,274]
df = pd.DataFrame({'volume' : volume, 'low' : low,'high' : high,'open' : open,'close' : close,'price' : price,'transactions' : transactions})
scaledf = sc.transform(df)
scaledx = np.array(scaledf[0:7,:])
scaledx = np.reshape(scaledx, (1, scaledx.shape[0], scaledx.shape[1]))
model.predict(scaledx)