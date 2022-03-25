from __future__ import print_function

import pandas as pd
import numpy as np
from gym import spaces
import gym
import time
import json
import pickle
def unscale(value, scaler, column):
    traindf_min = scaler.data_min_[column]
    traindf_max = scaler.data_max_[column]
    return ((value * (traindf_max - traindf_min)) + traindf_min)

def scale(value, scaler, column):
    traindf_min = scaler.data_min_[column]
    traindf_max = scaler.data_max_[column]
    return (value - traindf_min) / (traindf_max-traindf_min)


class UnscaledSimMarket(gym.Env):
    def __init__(self, cash = 0, data = []):
        super(UnscaledSimMarket, self).__init__()
        self.cash = cash
        self.start = time.time()
        self.initcash = self.cash
        self.coins = 0
        #start with only cash
        self.row = 0
        self.price = 0
        self.data = np.array(data)
        self.failstate = False
        
        #we will discretize to -10,10 here
        #self.action_space = spaces.Box(high = 1, low = -1, shape = (1,))
        self.action_space = spaces.Box(low= -1, high = 1, shape = (1,))
        high = np.full(np.shape(self.data[0]), 1)
        low = np.full(np.shape(self.data[0]), 0)
        #high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max])
        #low = np.array([0,0])
        self.observation_space = spaces.Box(np.float32(low),np.float32(high))
        self.state = None
        
        
        #per episode log
        self.actions = []
        self.vals = []
        self.roi = []
        
        
        #whole run log
        self.num_episodes = -1 #don't ask, logic really forces your hand sometimes, or laziness
        self.valperepisode = []
        self.actionperepisode = []
        self.roiperepisode = []
        self.endingvalperepisode = []
        
    
    #account value now
    def getAccountValue(self):
        return (self.price * self.coins) + self.cash

    #any input under 7 will look backward
    def getFutureAcctValue(self, lookforward):
        return (self.data[self.row+lookforward-7][1] * self.coins) + self.cash


    def getState(self):
        # if self.getAccountValue() < self.initcash/10:
        #     self.failState = True
        return np.array(self.state), self.rewardFunction(), self.failState, {}
    
    def getROI(self):
        return (self.getAccountValue() / self.initcash)
    
    def getDeltaValue(self, lookback):
        if len(self.valhistory) < lookback:
            return 0
        else:
            return self.getAccountValue() - self.valhistory[-lookback]
    
    def rewardFunction(self):
        if len(self.roi) == 0 or np.std(self.roi)== 0:
            return 0
        else:
            return ((self.roi[-1] - 0) / np.std(self.roi))
        
        #return ((self.roi[-1] - 0) / np.std(self.roi)) if (self.row - 1 != 0) else 0
        #return ((self.getROI() - 0) / np.std(self.acctvalue))
        # return (self.getAccountValue() - self.getPotentialGains())
        # return (self.getDeltaValue(7) / self.getAccountValue())
    
    def getPotentialGains(self):
        return (self.initcash / np.min(self.data[:,0][:self.row])) * np.max(self.data[:,0][:self.row])
        #return (self.initcash / np.min(self.data[:,0][0])) * np.max(self.data[:,0][:self.row])
    
    # def step(self, action):
    #     self.row += 1
    #     if self.row == 1800:
    #         return (0,0), 1, True, {}
    #     else:
    #         if action > 5 and action < 10:
    #             return np.array((1,1)), 1, False, {}
    #         else:
    #             return np.array((1,1)), 0, False, {}
    
    def step(self, action):
        #price,predprice
        if self.row >= len(self.data):
            self.failState = True
            # print(self.getAccountValue())
            print("Took ", time.time() - self.start ," to run. Account value at end of run: " , self.getAccountValue(), "difference from hodl point: ", (self.getAccountValue() - ((self.initcash / np.min(self.data[:,0])) * np.max(self.data[:,0])))) 
            return self.getState()
        self.state = self.data[self.row]
        #price
        self.price = self.state[0]
        self.failState = False
        action = action[0]
        #return np.append(self.data.to_numpy()[self.row-1],(self.acctvalue, self.coins, self.cash)), self.acctvalue, True
        #update market on each step
        
        #NN action is discrete, 0->20, so we have to convert
        #action = (action-10)/10
        #0<action<1  = buy coins
        #-1<action<0 = sell coins
        #0 = do nothing
        if action > 0 and action <= 1:
            #if self.price == 0 , unscale price, unscale cash, scale result
            self.coins += (self.cash * action) / self.price
            self.cash -= action * self.cash
            
        if action < 0 and action >=-1:
            self.cash -= (action * self.coins) * self.price
            self.coins += (action * self.coins)
            
        if action <-1 or action > 1:
            self.failState = True
            return self.getState()
        self.vals.append(self.getAccountValue())
        self.roi.append(self.getROI())
        self.actions.append(action)
        self.row += 1
        return self.getState()
    
    def render(self):
        return 1
    
    def reset(self):
        
        self.valperepisode.append(self.vals)
        self.actionperepisode.append(self.actions)
        self.roiperepisode.append(self.roi)
        self.endingvalperepisode.append(self.vals[-1:])
        self.num_episodes +=1
        
        self.row =0
        self.cash = self.initcash
        self.coins = 0
        self.price = 0
        self.start = time.time()
        self.failstate = False
        self.state = np.array(self.data[0])
        self.vals = []
        self.actions = []
        self.roi = []
        return self.state


from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """




    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _init_callback(self) -> None:
        self.rowlength = len(self.training_env.get_attr('data')[0])
    
    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        file = open('trainingdata.pkl','wb')
        numepisodes = self.training_env.get_attr('num_episodes')
        valperepisode = self.training_env.get_attr('valperepisode')
        actionperepisode = self.training_env.get_attr('actionperepisode')
        roiperepisode = self.training_env.get_attr('roiperepisode')
        endingvalperepisode = self.training_env.get_attr('endingvalperepisode')
        # jsonobj = {'episodes' : numepisodes, 'valperepisode' : valperepisode, 'actionperepisode' : actionperepisode,
        #            'roiperepisode' : roiperepisode, 'endingvalperepisode' : endingvalperepisode}
        # with open('training_data.json', 'w') as outfile:
        #     json.dump(jsonobj, outfile)
        pickle.dump(numepisodes, file)
        pickle.dump(valperepisode, file)
        pickle.dump(actionperepisode, file)
        pickle.dump(roiperepisode, file)
        pickle.dump(endingvalperepisode, file)
        file.close()
        pass
        
    def _on_step(self) -> bool:
        if(self.rowlength and (self.n_calls % self.rowlength) == 0):
            endingvaluelist = self.training_env.get_attr('endingvalperepisode')
            endingvaluelist = [endingvaluelist[-1:] for endingvaluelist  in endingvaluelist ]
            self.logger.record('endingval', np.mean(endingvaluelist))
            endingroi = self.training_env.get_attr('roiperepisode')
            endingroi = [endingroi[-1:] for endingroi  in endingroi ]
            self.logger.record('endingroi', np.mean(endingroi))
        return True
