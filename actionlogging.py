import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import seaborn as sns



# file = open("trainingdata.pkl", "rb")
# numepisodes = pickle.load(file)
# valperepisode = pickle.load(file)
# actionperepisode = pickle.load(file)
# roiperepisode = pickle.load(file)
# endingvalperepisode = pickle.load(file)
# file.close()

# plt.ioff()
# matplotlib.use('Agg')
# for i in range(1, len(actionperepisode[0])):
#     fig = plt.figure()
#     plt.scatter(range(len(actionperepisode[0][i])),actionperepisode[0][i], s= 0.5)
#     plt.title("Leverage throughout Training")
#     plt.xlabel("Episode Timesteps")
#     plt.ylabel("Leverage")
#     plt.savefig("./finalactualgraphs/trainingleverage/" + str(i) + ".png")
#     plt.close(fig)

# plt.ioff()
# matplotlib.use('Agg')
# for i in range(1, len(actionperepisode[0])):
#     fig, ax = plt.subplots()
#     x = [*range(len(actionperepisode[0][i]))]
#     y = actionperepisode[0][i]
#     plot = sns.jointplot(x=x, y=y, hue = 0, s = 5)
#     plot.set_axis_labels("Episode Timesteps", "Leverage")
#     plt.suptitle("Leverage throughout training")
#     plt.tight_layout()
#     plt.savefig("./finalactualgraphs/trainingleverage/" + str(i) + ".png")
#     print(i)
#     plt.close()
#     plt.clf()

# # # plt.scatter(range(15676),actionperepisode[0][1], s= 0.1)
# # # plt.scatter(range(15676),actionperepisode[0][1913], s= 0.1)
# # print('a')
# plt.plot(endingvalperepisode[0][1:])


# plt.ioff()
# matplotlib.use('Agg')
# for i in range(1, len(actionperepisode[0])):
#     sns.jointplot(x=[*range(len(actionperepisode[0][i]))], y=actionperepisode[0][i], kind = "kde")
#     plt.savefig("./figuresnew/hex/" + str(i) + ".png")
#     plt.clf()
#     plt.close()

# plt.scatter(range(15676),actionperepisode[0][1], s= 0.1)
# plt.scatter(range(15676),actionperepisode[0][1913], s= 0.1)
# print('a')
# # plt.plot(endingvalperepisode[0][1:])




# x = [*range(len(actionperepisode[0][1]))]
# y = actionperepisode[0][1]

# sns.jointplot(x=x, y=y, hue = 0, s = 5)
# plt.show()
# print("hi")


# #for making into gif:
# #ffmpeg -f image2 -framerate 10000 -i %01d.png -loop -1 scatterkde.gif



file = open('testingdataleveragefinallyactully.pkl','rb')
endings = pickle.load(file)
roiendings = np.divide(endings, 500)
fig, ax = plt.subplots()
ax.set_title("Validation Dataset Final ROI Through Training")
ax.set_xlabel("Episode")
ax.set_ylabel("Log Return on Investment")
ax.plot(roiendings)
plt.show()
file.close()
print("hi")
                
                




# model = PPO.load("ppohourly60percent_timesteps30000000_learningrate0.003_learnstarts5000_buffersize1000000_batchsize256_tau0.005_gamma0.97")

# obs = env.reset()
# done = False
# while done == False:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
# print('hi')



