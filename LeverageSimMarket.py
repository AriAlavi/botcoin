from __future__ import print_function

import pandas as pd
import numpy as np
from gym import spaces
import gym
import time
import json
import pickle
np.random.seed(42)
def unscale(value, scaler, column):
    traindf_min = scaler.data_min_[column]
    traindf_max = scaler.data_max_[column]
    return ((value * (traindf_max - traindf_min)) + traindf_min)

def scale(value, scaler, column):
    traindf_min = scaler.data_min_[column]
    traindf_max = scaler.data_max_[column]
    return (value - traindf_min) / (traindf_max-traindf_min)


class LeverageSimMarket(gym.Env):
    def __init__(self, cash = 0, data = []):
        super(LeverageSimMarket, self).__init__()
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
        self.action_space = spaces.Box(low= 0, high = 1, shape = (1,))
        high = np.full(np.shape(self.data[0]), 1)
        low = np.full(np.shape(self.data[0]), 0)
        #high = np.array([np.finfo(np.float32).max, np.finfo(np.float32).max])
        #low = np.array([0,0])
        self.observation_space = spaces.Box(np.float32(low),np.float32(high))
        self.state = None
        
        self.accountvalue = self.cash
        self.nowroi = 1
        
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

    def getState(self):
        # if self.getAccountValue() < self.initcash/10:
        #     self.failState = True
        return np.array(self.state), ((self.nowroi - 0) / np.std(self.roi)) if np.std(self.roi) != 0 else np.random.uniform(0,1), self.failState, {}
    
    def getROI(self):
        return (self.accountvalue / self.initcash)
    
    # def rewardFunction(self):
    #     return ((self.nowroi - 0) / np.std(self.nowroi))
    
    
    def step(self, action):
        #price,predprice
        self.failState = False
        if self.row >= len(self.data):
            self.failState = True
            # print(self.getAccountValue())
            print("Took ", np.round((time.time() - self.start),4) ," to run. Account value at end of run: " , np.round(self.accountvalue,2), "difference from hodl point: ", np.round((self.accountvalue - ((self.initcash / np.min(self.data[:,0])) * np.max(self.data[:,0]))),2), "  average action:" + str(np.mean(self.actions))) 
            return self.getState()
        self.state = self.data[self.row]
        #price
        self.price = self.state[0]
        action = action[0]
        
        # coinValue = self.coins * self.price
        # totalValue = self.cash + coinValue
        # currentLeverage = coinValue / totalValue
        
        # deltaLeverage = action - currentLeverage
        
        # deltaCash = deltaLeverage * totalValue
        # deltaCoins = deltaCash / self.price
        
        # self.cash = self.cash - deltaCash
        # self.coins = self.coins + deltaCoins
        
        oldcash = self.cash
        
        self.cash = (action - 1) * (-self.cash - (self.price*self.coins))
        self.coins = action * (oldcash/self.price + self.coins)
        
        
        self.accountvalue = self.getAccountValue()
        self.nowroi = self.getROI()
        self.vals.append(self.accountvalue)
        self.roi.append(self.nowroi)
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
        self.accountvalue = self.initcash
        self.nowroi = 1
        return self.state



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
