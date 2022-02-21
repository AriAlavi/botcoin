import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

scaler_filename = "trainscaler.save"
sc = joblib.load(scaler_filename) 
model = keras.models.load_model('400epoch64b20220214-143742shaved')

def unscale(value, scaler):
    traindf_min = scaler.data_min_[5]
    traindf_max = scaler.data_max_[5]
    return ((value * (traindf_max - traindf_min)) + traindf_min)


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


unscaled = scaleinputsandpredict(df)



print('hi')

