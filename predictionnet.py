import numpy as np
import pandas as pd
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
from sklearn.preprocessing import MinMaxScaler
import joblib
import time
import datetime

predahead = 36
#larger lookback values cause increasing validation
lookback = 36
pricecolindex = 7



dataframe = pd.read_excel("ethusdaligned.xlsx")
sentdf = pd.read_excel("hourlysentimentaligned.xlsx")
dataframe['sentiment'] = sentdf['Sentiment']
dataframe = dataframe.dropna().reset_index()
dataframe = dataframe.drop(dataframe.columns[0:4], axis = 1)
if(dataframe.isna().any().any() == True):
    quit()
#dataframe = dataframe.groupby(np.arange(len(dataframe))//3).mean()
name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "shaved"
sc = MinMaxScaler(feature_range = (0, 1))
# def signMomentum(sign):
#     momentum = []
#     mom = 0
#     LastSignPositive = True
#     for index, sign in enumerate(sign):
#         if sign == 0: #concern, we should probably count very small differences as 0 instead of increasing momentum
#             mom = 0
#         if sign == 1:
#             if LastSignPositive == True:
#                 mom += 1
#             else:
#                 mom = 1
#             LastSignPositive = True
#         if sign == -1:
#             if LastSignPositive == True:
#                 mom = -1
#             else:
#                 mom -= 1
#             LastSignPositive = False
#         if np.isnan(sign):
#             momentum.append(np.nan)
#         else:
#             momentum.append(mom)
#     return momentum
# dataframe['movingSentMean5'] = dataframe['safeMeanPrice'].rolling(5).sum()/5
# dataframe['movingSentMean10'] = dataframe['safeMeanPrice'].rolling(10).sum()/10
# dataframe['movingSentMean25'] = dataframe['safeMeanPrice'].rolling(25).sum()/25
# dataframe['movingSentMean50'] = dataframe['safeMeanPrice'].rolling(50).sum()/50
# dataframe['movingSentMean100'] = dataframe['safeMeanPrice'].rolling(100).sum()/100
# dataframe['movingSentMean500'] = dataframe['safeMeanPrice'].rolling(500).sum()/500


# dataframe['deltaPrice1Row'] = dataframe['price'].diff()
# dataframe['deltaPrice5Row'] = dataframe['price'].diff(periods=5)
# dataframe['deltaPrice10Row'] = dataframe['price'].diff(periods=10)
# dataframe['deltaPrice25Row'] = dataframe['price'].diff(periods=25)
# dataframe['deltaPrice50Row'] = dataframe['price'].diff(periods=50)
# dataframe['deltaPrice100Row'] = dataframe['price'].diff(periods=100)
# dataframe['deltaPrice200Row'] = dataframe['price'].diff(periods=200)
# dataframe['deltaPrice500Row'] = dataframe['price'].diff(periods=500)


# dataframe['deltaSign1Row'] = np.sign(dataframe['deltaPrice1Row'])
# dataframe['signMomentum1Row'] = signMomentum(dataframe['deltaSign1Row'])
# dataframe['deltaSign500Row'] = np.sign(dataframe['deltaPrice500Row'])
# dataframe['signMomentum500Row'] = signMomentum(dataframe['deltaSign500Row'])
# dataframe['std5Row'] = dataframe['price'].rolling(5).std()
# dataframe['std100Row'] = dataframe['price'].rolling(100).std()
# dataframe['volume5Row'] = dataframe['volume'].rolling(5).sum()
# dataframe['volume100Row'] = dataframe['volume'].rolling(100).sum()
# dataframe['volume500Row'] = dataframe['volume'].rolling(500).sum()

# dataframe['movingAverage5'] = dataframe['price'].rolling(5).sum()/5
# dataframe['movingAverage50'] = dataframe['price'].rolling(50).sum()/50
# dataframe['movingAverage500'] = dataframe['price'].rolling(500).sum()/500
# dataframe = dataframe.drop(dataframe.head(501).index)
training_set_scaled = sc.fit_transform(dataframe)
scaler_filename = "trainscaler.save"
joblib.dump(sc, scaler_filename + name) 

#split 80-20 training vs testing
train = np.array(training_set_scaled[:int(np.round(len(training_set_scaled)*.60))])
test = np.array(training_set_scaled[int(np.round(len(training_set_scaled)*.60)):])


def create_dataset(dataset, look_back, look_forward):
    x,y = [], []
    for i in range(len(dataset)):
        a = dataset[i:(i+look_back),:]
        if(i >= len(dataset) - look_back - 1):
            break
        x.append(a)
        y.append(dataset[i+1:i+look_forward+1,pricecolindex]) #last value is the column index for price, changes depending on spreadsheet
    return np.array(x), np.array(y)


def create_dataset_ahead(dataset, look_back, look_forward, skip_ahead = 0, pattern = 1):
    x,y = [], []
    for i in range(len(dataset)):
        a = dataset[i:(i+look_back),:]
        if(i >= len(dataset) - look_back - 1):
            break
        x.append(a)
        y.append(dataset[i+1+skip_ahead:i+look_forward+1:pattern,pricecolindex]) #last value is the column index for price, changes depending on spreadsheet
    test = dataset[:-37]
    if( x[len(x)-1][0][7] != test[len(test)-1][7] ):
        #important, as our data method returns a slightly smaller dataframe than the dataset
        print("frames not aligned, will cause bias and future knowledge")
        quit()
    return np.array(x), np.array(y)

def create_change_dataset(dataset, look_back, look_forward):
    x,y = [], []
    for i in range(len(dataset)):
        a = dataset[i:(i+look_back),:]
        if(i >= len(dataset) - look_back - 1):
            break
        x.append(a)
        val1 = dataset[0][pricecolindex]
        futurevalues = dataset[i+1:i+look_forward+1,pricecolindex]
        change = (futurevalues / val1) - 1
        y.append(change) #last value is the column index for price, changes depending on spreadsheet
    return np.array(x), np.array(y)




# xtrain,ytrain = create_dataset(train, lookback, predahead)
# xtest,ytest = create_dataset(test,  lookback, predahead)


xtrain,ytrain = create_dataset_ahead(train, lookback, predahead, skip_ahead = 0, pattern = 4)
xtest,ytest = create_dataset_ahead(test,  lookback, predahead, skip_ahead = 0, pattern = 4)




xtrain = np.reshape(xtrain, (xtrain.shape))
xtest = np.reshape(xtest, (xtest.shape))




log_dir = "logs/fit/" + name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)


model = Sequential()
model.add(LSTM(128, activation='tanh', recurrent_activation = 'sigmoid', recurrent_dropout= 0, unroll=False, use_bias = True, input_shape=(xtrain.shape[1],xtrain.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, activation='tanh', recurrent_activation = 'sigmoid', recurrent_dropout= 0, unroll=False, use_bias = True, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, activation='tanh', recurrent_activation = 'sigmoid', recurrent_dropout= 0, unroll=False, use_bias = True))
model.add(Dropout(0.2))
model.add(Dense(ytrain.shape[1]))
# model.add(Dense(predahead))
#VERY GOOD LEARNING RATE AT 0.00001 200 epochs in, 36 back 12 ahead, 0.002 loss quickly
#0.0015 60 epochs in, 36 back 36 ahead, very close loss/val at 200 epochs
model.compile(loss='mean_squared_error', optimizer=tf.optimizers.Adam(learning_rate=0.0001))
history = model.fit(xtrain, ytrain, validation_data=[xtest,ytest], callbacks=[tensorboard_callback], epochs=750, batch_size=256)
#model.save('100epoch256b-lr0.00001hourly7ahead' + name)


#this line above is where the model is trained and saved

#load the model

# # model = keras.models.load_model('400epoch64b' + name)
# model = keras.models.load_model('400epoch64b20220214-143742shaved')
#model = keras.models.load_model('100epoch256b-lr0.00001hourly7ahead20220301-022756shaved')
#predicting prices and stuffing them back into an array which can be scale inverted, total pain and complicated but required due to how normalization works
#I made a function to make this easier but its in another file

traindf_min = sc.data_min_
traindf_max = sc.data_min_

#7 is the index of the price volumn
def unscale(value, scaler):
    traindf_min = scaler.data_min_[pricecolindex]
    traindf_max = scaler.data_max_[pricecolindex]
    return ((value * (traindf_max - traindf_min)) + traindf_min)

def scale(value, scaler):
    traindf_min = scaler.data_min_[pricecolindex]
    traindf_max = scaler.data_max_[pricecolindex]
    return (value - traindf_min) / (traindf_max-traindf_min)


testmodel = model.predict(xtest)
testvalues = unscale(testmodel,sc)

trainmodel = model.predict(xtrain)
trainvalues = unscale(trainmodel,sc)


wholetest,wholetesty = create_dataset(training_set_scaled, lookback, predahead)
wholetest = np.reshape(wholetest, (wholetest.shape))
wholemodel = model.predict(wholetest)
wholevalues = unscale(wholemodel, sc)

testdf = pd.DataFrame(testvalues)
testpricedf = pd.DataFrame(dataframe['price'][int(np.round(len(dataframe)*.60)):]).reset_index()
testpricedf.drop(testpricedf.tail(predahead+1).index,inplace=True) # must remove last few values to align and have same size as the testvalues
testdf = testdf.divide(testpricedf['price'], axis=0) -1
testdf.insert(loc = 0, column = 'price', value = testpricedf['price'])

pd.DataFrame(testdf).to_excel("1hourly36hahead36hback60percenttest.xlsx") 

# wholedf = pd.DataFrame(wholevalues)
# pricedf = pd.DataFrame(dataframe['price']).reset_index()
# pricedf.drop(pricedf.tail(len(pricedf)-len(wholedf)).index,inplace=True)
# wholedf = wholedf.divide(pricedf['price'], axis=0) -1
# wholedf.insert(loc = 0, column = 'price', value = pricedf['price'])

# pd.DataFrame(wholedf).to_excel("1hourly36hahead36hback60percentall.xlsx")  

# #blue is predicted
# plt.plot(range(len(wholevalues)), wholevalues)
#  #orange is 
# plt.plot(range(len(dataframe['price'])), dataframe['price'])

# # plt.plot(range(0,len(testvalues)), tespltvalues)

# plt.show()