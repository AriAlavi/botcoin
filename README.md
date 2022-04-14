# botcoin

A comprehensive project for the discretizing of Kraken transaction data, sentiment analysis on historical Reddit submissions, the multivariate prediction of prices, and the simulation of trading for reinforcement a reinforcement learning algorithm. End goals were price prediction and trading simulation, current setup has 89% prediction accuracy and trading easily surpasses a human trader. If you wish to run any of this project, please see requirements at the bottom of this README

## Tools, Sources, and Specifications
* Pushshift.io Reddit Submission archive
* Kraken for Ethereum transaction data
* Tensorflow for the 3 layer LSTM prediction neural network
* Pytorch as a requirement for the trading neural network
* Stable-Baselines3, based on OpenAI's algorithms, for the Proximal Policy Optimization algorithm
* Tensorboard for logging of both networks
* 60-40 train-test split, chained across both networks. So 60% of the dataset is used for the training of the prediction network, and 60% of the remaining 40% for the trading network.

## Main
```main.py``` Is where we can access our Kraken discretization and output the data. After inputting the kraken data csv, we can output a dataframe in our chosen discrete timeframes, by default we have ``` long``` and ```short```, 1 hour and 24 hours respectively.

```predictionnet.py``` Here we we can predict future prices. By default we predict ```36``` hours in the future, using the past ```36``` hours of data. This data is first standarized and normalized, and after prediction is done we convert our prediction dataset to ```% change in price``` as well as shaving off neighboring points to improve on training speed for the reinforcement learning algorithm. Intuition tells us neighboring points can easily be extrapolated after removal, as we know the beginning, middle, and ends

```tradingnet.py``` With our predicted prices and our historical prices we can start to train a reinforcement learning network to trade. Being careful to have a correct train-test split(```60%-40%``` by default), training can start. This is an extremely cpu-intensive task, as our environment is long but not complicated, so the ```cpu``` flag is considerably faster. Depending on the hyperparameters of the model, training can happen very quickly or very slowly, but roughly takes at least 1 million timesteps.

```testtradingnet.py``` Here we can access the periodic model checkpoints, the training pickle, and work with the various logging variables for graphing and understanding

### Features

* Fetching and processing of our inputs for prediction, market data and reddit archives. 
* Customizable discrete ranges for our input data
* Functional Tensorboard logging for both prediction and trading models
* Hyperparameter searching and tuning for trading model
* Graphing tools for trading model results
* Lightweight and simple to understand training environment for the trading network. Includes Tensorboard callbacks and easy logging

### Requirements
Install our ```requirements.txt``` via pip, and for both networks if GPU(heavily recommended for prediction) is desired, see https://www.tensorflow.org/install/gpu
Some files are not included, such as the archives and our base data, but outputs are. See Pushshift.io and Kraken for reddit and transaction data
