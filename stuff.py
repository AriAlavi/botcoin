import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_excel("1hourly36hahead36hback60percenttestmse.xlsx")

if dataframe.isnull().values.any():
    print('na values exist')
    quit()


dataframe = dataframe.drop(['Unnamed: 0'], axis = 1)
print("hi")