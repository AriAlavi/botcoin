from pickletools import string1
from re import L
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import gym
import time
import pandas as pd
import datetime
import scipy.signal
import time
import LeverageSimMarket
import UnscaledSimMarket
import tensorflow as tf
from gym import spaces
import torch
import stable_baselines3 as baselines
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecCheckNan, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common import utils
import itertools
from typing import Callable

def linschedmodified(start: float, end: float, end_fraction: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction
    return func

def getSplit(filename, leftsplit):
    """
    Arguments:
        string: filename
        float: leftsplit, the train percentage. Testdf is 1-leftsplit
    Returns:
        dataframe: traindf, testdf
    """
    dataframe = pd.read_excel(filename)
    if dataframe.isnull().values.any():
        print('na values exist')
        quit()
    dataframe = dataframe.drop(['Unnamed: 0'], axis = 1)

    traindf = dataframe[:int(len(dataframe)*leftsplit)]
    testdf = dataframe[int(len(dataframe)*leftsplit):]
    return traindf, testdf

def parameterSearch(traindf):
    """
    Arguments:
        traindf: dataframe
    """
    #testdf = dataframe[int(len(dataframe)*.60):]
    parameters = [list(np.linspace(0.01,0.03,2)), [0.001],
                list(np.linspace(0.9999,0.95,3)),list(np.linspace(0.001,0.3,4)),
                list([32,2048])]

    #learningrate
    # ent_coef = [np.linspace(0.001, 0.2, 4)]
    # gamma = [np.linspace(0.9999,0.95,4)]
    # clip = [np.linspace(0.001,0.3,4)]
    # batch = [32,64,128,512,2048]
    combos = list(itertools.product(*parameters))
    i = 0
    while i < len(combos):
        model_parems = {'data' : traindf,
                    'timesteps'     : 150000,
                    'learningrate' : combos[i][0],
                    'ent_coef' : combos[i][1], #0.01
                    'gamma' : combos[i][2],
                    'policy' : "MlpPolicy", #MLPPOLICY DEFAULT
                    'clip'   : combos[i][3],
                    'batch_size' : combos[i][4]
        }
        values = list(model_parems.values())
        string = ""
        for key,value in model_parems.items():
            if key == 'data':
                string += 'minileveragesearch'
                continue
            string += "_{}{}".format(key,value)
        env = LeverageSimMarket.LeverageSimMarket
        env = make_vec_env(env, n_envs=1, env_kwargs = {'cash' : 500, 'data' : values[0]}, seed=42)
        env = VecCheckNan(env, raise_exception=True)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1., clip_reward=1.)

        #checkpoint_callback = CheckpointCallback(save_freq=len(traindf), save_path='./' + str(string) + 'checkpoint/',
        #                                        name_prefix='ppo_model')
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log = 'minitesting2', seed=42, learning_rate= values[2], device = "cpu", ent_coef = values[3], gamma=values[4], clip_range=values[6], batch_size = values[7]) #significantly faster on cpu
        callback= LeverageSimMarket.TensorboardCallback()

        # model = PPO.load("ppohourly60trainnewmse_timesteps5000000_learningrate0.003", env = env, device = "cpu")
        model.learn(total_timesteps=values[1], log_interval=1, callback = [callback], tb_log_name =string, reset_num_timesteps= False)
        model.save("/minitesting/" + string)
        del env
        del string
        del model
        del callback
        del model_parems
        i += 1



def vecEnvTry(parems, tbfolder, numvecs, learningrate = None, schedlist = None):
    parems_ = parems
    parems = list(parems.values())
    env = LeverageSimMarket.LeverageSimMarket
    # check_env(env, warn = True, skip_render_check = True)
    env = make_vec_env(env, n_envs=numvecs, env_kwargs = {'cash' : 500, 'data' : parems[0]}, seed=parems[2], vec_env_cls=SubprocVecEnv)
    env = VecCheckNan(env, raise_exception=True)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1., clip_reward=1.)
    numvecs = 6
    string = ""
    for key,value in parems_.items():
        if key == 'data':
            string += 'postfinal'
            continue
        string += "_{}{}".format(key,value)
    if learningrate is float:
        string+= 'learningrate' + learningrate
    elif isinstance(schedlist, list):
        string += 'lrbeg' + str(schedlist[0]) + 'lrend' + str(schedlist[1]) + 'frac' + str(schedlist[2])
        learningrate =  utils.get_linear_fn(schedlist[0],schedlist[1],schedlist[2])
    if learningrate == None and schedlist == None:
        'You need to include a learning rate'
        quit()
    checkpoint_callback = CheckpointCallback(save_freq=len(parems[0]), save_path='./' + str(string) + 'defaultcheckpoint/',
                                            name_prefix='postfinal')
    #model = PPO("MlpPolicy", env, verbose=0, tensorboard_log = 'results', seed=42, learning_rate= values[2], device = "cpu", ent_coef = values[3], gamma=values[4], clip_range=values[6], batch_size = values[7]) #significantly faster on cpu
    if type(learningrate) is float:
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log = tbfolder, seed=parems[2], learning_rate= learningrate, device = "cpu", gamma=parems[4], clip_range=parems[6], ent_coef=parems[3], batch_size=2048) #significantly faster on cpu
    else:
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log = tbfolder, seed=parems[2], learning_rate= learningrate, device = "cpu", gamma=parems[4], clip_range=parems[6], ent_coef=parems[3]) #significantly faster on cpu
    callback= LeverageSimMarket.TensorboardCallback()
    # model = PPO.load("ppohourly60trainnewmse_timesteps5000000_learningrate0.003", env = env, device = "cpu")
    model.learn(total_timesteps=parems[1], log_interval=1, callback = [callback, checkpoint_callback], tb_log_name = string, reset_num_timesteps= False)
    model.save(string)


def makeEnv(parems):
    parems = list(parems.values())
    env = LeverageSimMarket.LeverageSimMarket
    # check_env(env, warn = True, skip_render_check = True)
    env = make_vec_env(env, n_envs=1, env_kwargs = {'cash' : 500, 'data' : parems[0]}, seed=parems[2])
    env = VecCheckNan(env, raise_exception=True)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1., clip_reward=1.)
    return env

def Train(parems, tbfolder, learningrate = None, schedlist = None):
    string = ""
    for key,value in parems.items():
        if key == 'data':
            string += 'postfinal'
            continue
        string += "_{}{}".format(key,value)
    if isinstance(learningrate, float):
        string+= 'learningrate' + str(learningrate)
    elif isinstance(schedlist, list):
        string += 'lrbeg' + str(schedlist[0]) + 'lrend' + str(schedlist[1]) + 'frac' + str(schedlist[2])
        learningrate =  linschedmodified(schedlist[0],schedlist[1],schedlist[2])
    if learningrate == None and schedlist == None:
        'You need to include a learning rate'
        quit()
    parems = list(parems.values())
    checkpoint_callback = CheckpointCallback(save_freq=len(parems[0]), save_path='./' + str(string) + 'defaultcheckpoint/',
                                            name_prefix='postfinal')
    #model = PPO("MlpPolicy", env, verbose=0, tensorboard_log = 'results', seed=42, learning_rate= values[2], device = "cpu", ent_coef = values[3], gamma=values[4], clip_range=values[6], batch_size = values[7]) #significantly faster on cpu
    if isinstance(learningrate, float):
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log = tbfolder, seed=parems[2], learning_rate= learningrate, device = "cpu", gamma=parems[4], clip_range=parems[6], ent_coef=parems[3]) #significantly faster on cpu
    else:
        model = PPO("MlpPolicy", env, verbose=0, tensorboard_log = tbfolder, seed=parems[2], learning_rate= learningrate, device = "cpu", gamma=parems[4], clip_range=parems[6], ent_coef=parems[3]) #significantly faster on cpu
    callback= LeverageSimMarket.TensorboardCallback()
    # model = PPO.load("ppohourly60trainnewmse_timesteps5000000_learningrate0.003", env = env, device = "cpu")
    model.learn(total_timesteps=parems[1], log_interval=1, callback = [callback, checkpoint_callback], tb_log_name = string, reset_num_timesteps= False)
    model.save(string)
    








if __name__ == '__main__':
    
    #best timesteps 1000000
    #seed 41
    #ent 0.0001
    #gamma 0.9999
    #clip 0.2
    #[0.2,0.000001,.60]
    np.random.seed(int(time.time()))
    train, test = getSplit("1hourly36hahead36hback60percenttest.xlsx", 0.6)
    parems = {
        'data' : train,
        'timesteps'  : 1000000,
        'seed' : 41,
        'ent_coef' : 0.0001, #0.01
        'gamma' : 0.9999,
        'policy' : "bestfinal", #MLPPOLICY DEFAULT
        'clip'   : 0.2,
    }
    
    env = makeEnv(parems)
    Train(parems, schedlist = [0.2,0.000001,.60], tbfolder = "tblogging")
    

    


#notes:
#default is bad
#
# values = list(model_parems.values())
# string = ""
# for key,value in model_parems.items():
#     if key == 'data':
#         string += 'postfinal'
#         continue
#     string += "_{}{}".format(key,value)
    
# env = UnscaledSimMarket.UnscaledSimMarket
# # check_env(env, warn = True, skip_render_check = True)
# env = make_vec_env(env, n_envs=1, env_kwargs = {'cash' : 500, 'data' : values[0]}, seed=42)
# env = VecCheckNan(env, raise_exception=True)
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=1., clip_reward=1.)

# checkpoint_callback = CheckpointCallback(save_freq=len(traindf), save_path='./' + str(string) + 'defaultcheckpoint/',
#                                          name_prefix='postfinal')
# #model = PPO("MlpPolicy", env, verbose=0, tensorboard_log = 'results', seed=42, learning_rate= values[2], device = "cpu", ent_coef = values[3], gamma=values[4], clip_range=values[6], batch_size = values[7]) #significantly faster on cpu
# model = PPO("MlpPolicy", env, verbose=0, tensorboard_log = 'postfinal', seed=42, learning_rate= utils.get_linear_fn(0.2,0.000001,.60), device = "cpu", gamma=values[4], clip_range=values[6], ent_coef=values[3]) #significantly faster on cpu
# callback= LeverageSimMarket.TensorboardCallback()
# # model = PPO.load("ppohourly60trainnewmse_timesteps5000000_learningrate0.003", env = env, device = "cpu")
# model.learn(total_timesteps=values[1], log_interval=1, callback = [callback, checkpoint_callback], tb_log_name = string, reset_num_timesteps= False)
# model.save(string)
    
    
    
#BEST AND STABLE!!!@@@@@
# train, test = getSplit("1hourly36hahead36hback60percenttest.xlsx", 0.6)
# parems = {
#     'data' : train,
#     'timesteps'     : 2000001,
#     'seed' : 42,
#     'ent_coef' : 0.001, #0.01
#     'gamma' : 0.99999,
#     'policy' : "MlpPolicy", #MLPPOLICY DEFAULT
#     'clip'   : 0.2,
# }
# #    utils.get_linear_fn(0.2,0.000001,.60)
# env = makeEnv(parems)
# #run again, then try it again on a different seed
# Train(parems, learningrate = None, schedlist = [0.2,0.000001,.60], tbfolder = "postfinal")
# print('hi')