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

dataframe = pd.read_excel("1hourly36hahead36hback60percenttest.xlsx")

if dataframe.isnull().values.any():
    print('na values exist')
    quit()


dataframe = dataframe.drop(['Unnamed: 0'], axis = 1)

testdf = dataframe[int(len(dataframe)*.60):]

env = UnscaledSimMarket.UnscaledSimMarket
# check_env(env, warn = True, skip_render_check = True)
env = make_vec_env(env, n_envs=1, env_kwargs = {'cash' : 500, 'data' : testdf})
env = VecCheckNan(env, raise_exception=True)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1., clip_reward=1.)

model = PPO.load("ppohourly60percent_timesteps30000000_learningrate0.003_learnstarts5000_buffersize1000000_batchsize256_tau0.005_gamma0.97")

obs = env.reset()
done = False
while done == False:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
print('hi')