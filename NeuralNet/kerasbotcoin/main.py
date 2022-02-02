import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from keras.layers import Normalization
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Input

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


import SimMarket

#This is for our datasets for keras/tensorflow, here I use it to create our price prediction dataset.
#One col is for the future price, rest are inputs such as current price, mean, etc
import pandas as pd
#since short is hourly, 24 rows would be 1 day in the future. 7 rows for long is 1 week
def addfuturepricecol(filename, rows):
    dataframe = pd.read_excel(filename)
    dataframe.iloc[:,0] = dataframe['price'].shift(-rows)
    dataframe = dataframe.rename(columns = {'Unnamed: 0' : 'futurePrice'})
    #remove last n rows because they have nofuture price
    dataframe.drop(dataframe.tail(rows).index,inplace=True)
    dataframe.drop(dataframe.head(rows).index,inplace=True)
    dataframe.drop('endDate', axis = 1, inplace=True)
    dataframe.drop('date', axis = 1, inplace=True)
    dataframe.drop(['std100Row','movingAverage50'],axis=1, inplace=True)
    return dataframe
    

dataframe = addfuturepricecol("ethusdlongtest.xlsx", 7)
dataframe.drop(['futurePrice'],axis=1, inplace=True)


sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(dataframe)

train = np.array(training_set_scaled[:1400])
test = np.array(training_set_scaled[1401:])

def create_dataset(dataset, look_back):
    x,y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back),:]
        x.append(a)
        y.append(dataset[i+look_back,5])
    return np.array(x), np.array(y)

xtrain,ytrain = create_dataset(train, 7)
xtest,ytest = create_dataset(test,  7)

savedxtrain = xtrain
savedxtest = xtest

xtrain = np.reshape(xtrain, (xtrain.shape[0], 7,14))
#ytrain = np.reshape(ytrain, (ytrain.shape[0], 7,14))



xtest = np.reshape(xtest, (xtest.shape[0], 7,14))
#ytest = np.reshape(ytest, (ytest.shape[0], 7,14))



model = Sequential()
model.add(LSTM(30, input_shape=(7,14), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(30, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=30))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(xtrain, ytrain, validation_split=.33, epochs=50, batch_size=32)


#features are the inputs, returns prediction for other uses
def invertPredictionAndPlot(prediction, features):
    values = np.zeros((len(prediction),14))
    values[:,5] = prediction[:,0]
    values = sc.inverse_transform(values)[:,5]
    plt.plot(range(values), values)
    return values



def plotprediction(prediction):
    plt.plot(range(len(prediction)), prediction)


#predicting prices and stuffing them back into an array which can be scale inverted
testmodel = model.predict(xtest)
testvalues = np.zeros((len(testmodel),14))
testvalues[:,5] = testmodel[:,0]
testvalues = sc.inverse_transform(testvalues)[:,5]


trainmodel = model.predict(xtrain)
trainvalues = np.zeros((len(trainmodel),14))
trainvalues[:,5] = trainmodel[:,0]
trainvalues = sc.inverse_transform(trainvalues)[:,5]

wholetest,wholetesty = create_dataset(training_set_scaled, 7)
wholetest = np.reshape(wholetest, (wholetest.shape[0], 7, 14))
wholemodel = model.predict(wholetest)
wholevalues = np.zeros((len(wholemodel),14))
wholevalues[:,5] = wholemodel[:,0]
wholevalues = sc.inverse_transform(wholevalues)[:,5]
wholevalues


#blue
plt.plot(range(len(wholevalues)), wholevalues)
#orange
plt.plot(range(len(dataframe['price'])), dataframe['price'])

plt.plot(range(0,len(testvalues)), testvalues)

plt.show()

print('hi')



dataframe['predictedprice7'] = np.pad(wholevalues, (0,len(dataframe)-len(wholevalues)))


# normalizer = Normalization(axis=-1)

# normalizer.adapt(train)

# normalized_data = normalizer(np.array(train))


# y_train = normalized_data['futurePrice']
# x_train = normalized_data.iloc[:,1:]


qdf = pd.DataFrame({'price':dataframe['price'].to_numpy(), 'predicted7' : dataframe['predictedprice7'].to_numpy()})

#try Q learning now
seed = 42
gamma = 0.99
max_steps_per_episode = len(dataframe)
eps = np.finfo(np.float32).eps.item()


num_inputs = 2
num_actions = 2
num_hidden = 128
inputs = Input(shape = (num_inputs,))
common = Dense(num_hidden, activation="relu")(inputs)
action = Dense(num_actions, activation="softmax")(common)
critic = Dense(1)(common)
qmodel = keras.Model(inputs = inputs, outputs = [action, critic])

#train

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

s = np.random.uniform(-1,1,1)

#consider making simmarket start point random
while True:
    sim = SimMarket.SimMarket(cash = 500, data = qdf)
    #try tuple
    state = sim.getInputs()
    episode_reward = 0
    with tf.GradientTape() as tape:
        #loop through sim
        for step in range(max_steps_per_episode):
            
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state,0)
            
            action_probs,critic_value = qmodel(state)
            critic_value_history.append(critic_value[0,0])
            
            #action = np.random.uniform(-1,1,1)[0]
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(action_probs[0,action]))
            #state might need tuple, as it is just an array
            state, acctvalue, failstate = sim.step(action)
            rewards_history.append(acctvalue)
            episode_reward = acctvalue
            
            if failstate:
                break
        
        running_reward = 0.05 * episode_reward + (1-0.05) * running_reward
        
        
        
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        returns = np.array(returns)
        returns = (returns-np.mean(returns)) / (np.std(returns)+eps)
        returns = returns.tolist()
        
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        
        for log_prob,value,ret in history:
            diff = ret-value
            actor_losses.append(-log_prob*diff)
            
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )
            
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, qmodel.trainable_variables)
        optimizer.apply_gradients(zip(grads, qmodel.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 100000000:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break