import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataframe = pd.read_excel("1hourly36hahead36hback60percenttestmse.xlsx")
testdf = dataframe[int(len(dataframe)*0.6):]
traindf = dataframe[:int(len(dataframe)*0.6)]
plt.title("Validation Dataset Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.plot(testdf['price'])





dataframe = pd.read_excel("1hourly36hahead36hback60percenttestmse.xlsx")
plt.plot(dataframe['price'], alpha=0.6, label='True Price')
plt.plot(dataframe[0]*(dataframe['price']) + dataframe['price'], alpha=0.6, label = '4 hours ahead')
plt.plot(dataframe[8]*(dataframe['price']) + dataframe['price'], alpha=0.6, label = '36 hours ahead')
plt.legend()

plt.title("Log-True Price and Log-Predicted Prices")
plt.xlabel("Hours")
plt.ylabel("USD")
print("hi")


dataframe = pd.read_excel("hourlysentimentaligned.xlsx")
plt.scatter(range(len(dataframe['Sentiment'])),dataframe['Sentiment'], s=0.5)
plt.title("Reddit Sentiment Regarding Crypto")
plt.xlabel("Time")
plt.ylabel("Sentiment")
print(' hi')