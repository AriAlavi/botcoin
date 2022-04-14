from pickletools import string1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from gym import spaces
import gym
import tensorflow as tf
import time
import re
import LeverageSimMarket
import UnscaledSimMarket
import pandas as pd
import datetime
import scipy.signal
import time
import torch
import stable_baselines3 as baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import pickle
import seaborn as sns


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataframe = pd.read_excel("1hourly36hahead36hback60percenttestmse.xlsx")

if dataframe.isnull().values.any():
    print('na values exist')
    quit()


dataframe = dataframe.drop(['Unnamed: 0'], axis = 1)

testdf = dataframe[int(len(dataframe)*0.6):]

env = LeverageSimMarket.LeverageSimMarket
# check_env(env, warn = True, skip_render_check = True)
env = make_vec_env(env, n_envs=1, env_kwargs = {'cash' : 500, 'data' : testdf})
env = VecCheckNan(env, raise_exception=True)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1., clip_reward=1.)



# directory = 'postfinal_timesteps1000000_seed41_ent_coef0.0001_gamma0.9999_policylastbestfinalpls_clip0.2lrbeg0.2lrend1e-06frac0.6'
# def nat_key(value):
#     return tuple(int(s) if s.isdigit() else s for s in re.split("(\d+)",value ))
# ls = sorted(os.scandir(directory), key=lambda e: nat_key(e.name))
# # ls = ls[-20:]
# valhist = []
# actionlist = []
# for filename in ls:
#     if filename.is_file():
#         model = PPO.load(filename.path, device="cpu")
#         obs = env.reset()
#         done = False
#         while done == False:
#             action, _states = model.predict(obs, deterministic=True)
#             obs, reward, done, info = env.step(action)
#             if done == True:
#                 valhist.append(env.get_attr("endingvalperepisode")[-1][-1][0])
#                 actionlist.append(env.get_attr("actionperepisode")[-1][-1])
                
                
                
# file = open('testingdataleveragefinallyactully.pkl','wb')
# pickle.dump(valhist, file)
# pickle.dump(actionlist,file)
# file.close()
# print("hi")
                

#gif is of training actions
#

#load actions on final model and graph

model = PPO.load("postfinal_timesteps1000000_seed41_ent_coef0.0001_gamma0.9999_policylastbestfinalpls_clip0.2lrbeg0.2lrend1e-06frac0.6", device = "cpu")
obs = env.reset()
actions = []
done = False
while done == False:
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, info = env.step(action)
    if done == True:
        actions.append(env.get_attr("actionperepisode")[0][1])

fig, ax = plt.subplots()


x = [*range(len(actions[0]))]
y = actions[0]

sns.jointplot(x=x, y=y, hue = 0, s = 5)
plt.show()

ax.set_title("Final Model Leverage on Validation over Time")
ax.set_xlabel("Validation Timesteps")
ax.set_ylabel("Leverage")
ax.plot(actions[0])
print("hi")


dataframe = pd.read_excel("1hourly36hahead36hback60percenttest.xlsx")

if dataframe.isnull().values.any():
    print('na values exist')
    quit()


dataframe = dataframe.drop(['Unnamed: 0'], axis = 1)

testdf = dataframe[int(len(dataframe)*.60):]
copydf = testdf.copy()
copydf.price = copydf.price*0
copydf.price = 5

env = LeverageSimMarket.LeverageSimMarket
# check_env(env, warn = True, skip_render_check = True)
env = make_vec_env(env, n_envs=1, env_kwargs = {'cash' : 500, 'data' : copydf})
env = VecCheckNan(env, raise_exception=True)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1., clip_reward=1.)
obs = env.reset()
actions = []
done = False
while done == False:
    action = []
    obs, reward, done, info = env.step([[np.random.uniform()]])
    if done == True:
        actions.append(env.get_attr("actionperepisode")[0][1])

# fig, ax = plt.subplots()
# ax.set_title("Final Model Leverage on Validation over Time")
# ax.set_xlabel("Validation Timesteps")
# ax.set_ylabel("Leverage")
# ax.plot(actions[0])
# print("hi")