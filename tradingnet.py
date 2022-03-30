from pickletools import string1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from gym import spaces
import gym
import tensorflow as tf
import time
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
from stable_baselines3.common.callbacks import EvalCallback

#cleaning
# dataframe = pd.read_excel("20172021ethshortaligned2017-21.xlsx")
# dataframe = dataframe.groupby(np.arange(len(dataframe))//4).mean()
# dataframe = dataframe.drop(dataframe.head(501).index)
# pred = pd.read_excel("4hourly48hahead48hback2500epoch.xlsx")
# dataframe.drop(dataframe.tail(len(dataframe)-len(pred)).index, inplace=True)
# dataframe = pd.concat([pd.DataFrame(data = {'price' : dataframe['price']}), pred],axis=1)
dataframe = pd.read_excel("1hourly36hahead36hback60percenttest.xlsx")

if dataframe.isnull().values.any():
    print('na values exist')
    quit()


dataframe = dataframe.drop(['Unnamed: 0'], axis = 1)

traindf = dataframe[:int(len(dataframe)*60)]
testdf = dataframe[int(len(dataframe)*.60):]

#defaults
#if you change data change the lower string pls for logging
model_parems = {'data' : traindf,
                'timesteps'     : 30000000,
                'learningrate' : 0.003,
                'learnstarts'   : 5000,
                'buffersize'   : 1000000,
                'batchsize'    : 256,
                'tau'           : 0.005,
                'gamma'         : .97}

values = list(model_parems.values())
string = ""
for key,value in model_parems.items():
    if key == 'data':
        string += 'ppohourly60percent'
        continue
    string += "_{}{}".format(key,value)
    
env = UnscaledSimMarket.UnscaledSimMarket
# check_env(env, warn = True, skip_render_check = True)
env = make_vec_env(env, n_envs=1, env_kwargs = {'cash' : 500, 'data' : values[0]}, seed=42)
env = VecCheckNan(env, raise_exception=True)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1., clip_reward=1.)


model = PPO("MlpPolicy", env, verbose=0, tensorboard_log = 'ppohourlylog30percent', seed=42, learning_rate= 0.0003)
callback= UnscaledSimMarket.TensorboardCallback()
model.learn(total_timesteps=values[1], log_interval=1, callback = [callback], tb_log_name = string)

model.save(string)


# model = PPO.load("ppohourly30percent_timesteps200000_learningrate0.003_learnstarts5000_buffersize1000000_batchsize256_tau0.005_gamma0.94", env = env)
# model.learn(total_timesteps=values[1], log_interval=1, callback = [callback], tb_log_name = string)
# model.save(string)
# env = UnscaledSimMarket.UnscaledSimMarket(cash = 500, data = testdf)
# obs = env.reset()
# done = False
# while done == False:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
# print('hi')

# def test(times):
#     obs = env.reset()
#     done = False
#     while done == False:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)
# print()