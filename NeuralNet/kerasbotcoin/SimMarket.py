from __future__ import print_function
import neat
import pandas
import numpy as np
import sklearn.preprocessing as preprocessing
import SimMarket

class SimMarket():
    def __init__(self, cash = 0, data = []):
        self.cash = cash
        self.coins = 0
        #start with only cash
        self.acctvalue = cash
        self.row = 0
        self.price = 0
        self.data = data
        self.failstate = False
    # def step(self, action):
    #     if self.row > len(self.price):
    #         raise NameError('Market sim exceeded data')
    #     #update market on each step
    #     self.price = self.price[self.row]
    #     #0<action<1  = buy coins
    #     #-1<action<0 = sell coins
    #     #0 = do nothing
    #     if action > 0 and action <= 1:
    #         self.coins += (self.cash * action) / self.price
    #         self.cash -= action * self.cash
    #     if action < 0 and action <=-1:
    #         self.cash -= (action * self.coins) * self.price
    #         self.coins += (action * self.coins)
    #     if action <-1 or action > 1:
    #         self.failState = True
    #         self.acctValue = 0
    #         return
    #     self.acctvalue = self.cash + (self.price * self.coins)
    #     self.row += 1
    def getInputs(self):
        return self.data.iloc[self.row]
    def getValue(self):
        return self.acctvalue
    def step(self, action):
        if self.row > len(self.data):
            raise NameError('Market sim exceeded data')
        #update market on each step
        self.price = self.data['price'][self.row]
        #0<action<1  = buy coins
        #-1<action<0 = sell coins
        #0 = do nothing
        if action > 0 and action <= 1:
            self.coins += (self.cash * action) / self.price
            self.cash -= action * self.cash
        if action < 0 and action <=-1:
            self.cash -= (action * self.coins) * self.price
            self.coins += (action * self.coins)
        if action <-1 or action > 1:
            self.failState = True
            # self.acctValue = 0
            self.acctvalue = self.cash + (self.price * self.coins)
            return
        self.acctvalue = self.cash + (self.price * self.coins)
        self.row += 1
        return self.data.iloc[self.row-1], self.acctvalue, self.failstate


# dataframe = pandas.read_excel("bigdataframe.xlsx")
# arr = np.delete(dataframe.to_numpy(),[0,1,2,3],1)
# arr = np.delete(arr, [range(0,1000)], 0)
# maxabs = preprocessing.MaxAbsScaler()
# scaledinput = maxabs.fit_transform(arr)


# inputs = scaledinput




# sim = SimMarket(500, 0, inputs)
# sim.step(.5)
# sim.step(-.5)
# sim.step(2)
# print("hello")