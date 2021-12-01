import csv
from datetime import date, datetime, timedelta
from multiprocessing import Pool
from decimal import Decimal
import multiprocessing
import pickle
import pathlib
import os

import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd


from dataTypes import *
import hypothesis


class RawData:
    def __init__(self, filename):
        self.filename = filename
        self.data = self.readFile()

    def readFile(self):
        assert isinstance(self.filename, str)
        file = open(self.filename)
        reader = csv.reader(file)
        i = 0
        DATA_COLLECTION = []
        for row in reader:
            DATA_COLLECTION.append(DataPoint(int(row[0]), float(row[1]), float(row[2])))

        file.close()
        print("File {} read".format(self.filename))
        return DATA_COLLECTION
    
    def fetchData(self, givenDate, givenWindow):
        assert isinstance(givenDate, datetime)
        assert isinstance(givenWindow, timedelta)
        FETCHED_DATA = []

        endDate = givenDate + givenWindow
        endDateInt = int(endDate.timestamp())
        
        for transaction in self.data:
            if  givenDate <= datetime.fromtimestamp(transaction.time) <= endDate:
                FETCHED_DATA.append(transaction)
            if transaction.time > endDateInt:
                break

        return FETCHED_DATA


def setLeverage(myCash, myCoins, coinPrice, soughtLeverage):
    assert isinstance(myCash, Decimal)
    assert isinstance(myCoins, Decimal)
    assert isinstance(coinPrice, Decimal)
    assert isinstance(soughtLeverage, Decimal)
    assert Decimal("0") <= soughtLeverage <= Decimal("1"), "was {} instead".format(float(soughtLeverage))
    assert myCash >= 0, "Was {} instead".format(myCash)
    assert myCoins >= 0

    if coinPrice == None or coinPrice <= 0 or myCash + myCoins == 0:
        return {
            "cash" : myCash,
            "coins" : myCoins
        }

    coinValue = Decimal(myCoins * coinPrice)
    totalValue = Decimal(myCash + coinValue)
    assert totalValue > 0
    currentLeverage = Decimal(coinValue) / Decimal(totalValue)

    leverageDelta = soughtLeverage-currentLeverage

    newCash = myCash
    newCoins = myCoins

    # print("Delta", float(leverageDelta))

    if leverageDelta > 0:
        toSpendOnCoins = leverageDelta * totalValue
        coinsPurchased = toSpendOnCoins / coinPrice
        newCash = myCash - toSpendOnCoins
        newCoins = myCoins + coinsPurchased
    elif leverageDelta < 0:
        toSellCoins = leverageDelta * totalValue * -1
        coinsSold = toSellCoins / coinPrice
        newCash = myCash + toSellCoins
        newCoins = myCoins - coinsSold

    return {
        "cash" : newCash,
        "coins" : newCoins
    }


def simulation(startingDate, timeSteps, endingDate, shortTermData, longtermData, hypothesisFunc, startingCash):
    assert isinstance(startingDate, datetime) # Should be the exact same starting date as the short term and long term data
    assert isinstance(timeSteps, timedelta) # Should be the exact same as the time window for the short term data
    assert isinstance(endingDate, datetime) # Should be the exact same as the starting date + window length for the fetched data
    assert isinstance(shortTermData, list)
    assert isinstance(longtermData, list)
    assert all(isinstance(x, DiscreteData) for x in shortTermData)
    assert all(isinstance(x, DiscreteData) for x in longtermData)
    assert len(shortTermData) > 0
    assert len(longtermData) > 0
    assert callable(hypothesisFunc)
    assert isinstance(startingCash, Decimal)
    
    assert startingDate < endingDate

    CASH = startingCash
    BOTCOINS = Decimal(0)
    BOTCOIN_PRICE = Decimal(0)
    VALUE_HISTORY = []
    LEVERAGE_HISTORY = []
    DATETIME_HISTORY = []
    CHARTING_PARAMETERS_HISTORY = {}

    now = startingDate
    

    MAX_SHORT_TERM_INDEX = len(shortTermData) - 1
    MAX_LONG_TERM_INDEX = len(longtermData) - 1
    shortTermIndex = -2
    longTermIndex = 0

    LONG_TERM_BEGINS = longtermData[longTermIndex].endDate
    while now <= LONG_TERM_BEGINS:
        now += timeSteps
        shortTermIndex += 1

    epsilon = Decimal(.0000000001)

    customParameters = {}
    while now < endingDate:
        if shortTermIndex < MAX_SHORT_TERM_INDEX:
            shortTermIndex += 1
        else:
            print("It is {} and the index is {} and the date of that index is {}. This shouldn't be possible, but I'm going to pretend like nothing has gone wrong.".format(now, shortTermIndex, shortTermData[shortTermIndex].date))

        
        currentShortTerm = shortTermData[shortTermIndex] 
        currentLongTerm = longtermData[longTermIndex]

        assert now == currentShortTerm.endDate, "Simulation is at {} but short term window is at {}".format(now, currentShortTerm.endDate)

        if currentShortTerm.safeMeanPrice:
            BOTCOIN_PRICE = Decimal(currentShortTerm.safeMeanPrice)

        if longTermIndex < MAX_LONG_TERM_INDEX:
            breaking = False
            nextLongTerm = longtermData[longTermIndex+1]
            while nextLongTerm.endDate <= now and not breaking:
                currentLongTerm = nextLongTerm
                longTermIndex += 1
                try:
                    nextLongTerm = longtermData[longTermIndex+1]
                except:
                    breaking = True

        chartingParameters = {}
        soughtLeverage = hypothesisFunc(currentShortTerm, currentLongTerm, CASH, BOTCOINS, customParameters, chartingParameters)
        if len(chartingParameters.keys()) > 0:
            if len(CHARTING_PARAMETERS_HISTORY.keys()) == 0:
                for key, value in chartingParameters.items():
                    CHARTING_PARAMETERS_HISTORY[key] = [value,]
            else:
                for key, value in chartingParameters.items():
                    CHARTING_PARAMETERS_HISTORY[key].append(value)

        newLeverage = setLeverage(CASH, BOTCOINS, BOTCOIN_PRICE, soughtLeverage)
        CASH = newLeverage["cash"]
        BOTCOINS = newLeverage["coins"]
        if CASH < 0:
            if CASH + epsilon < 0:
                raise Exception("Cash cannot go negative! It is {}".format(CASH))
            else:
                # print("Epsilon problem encountered for cash: {}".format(CASH))
                CASH = Decimal(0)

        if BOTCOINS < 0:
            if BOTCOINS + epsilon < 0:
                raise Exception("Botcoins cannot go negative! It is {}".format(BOTCOINS))
            else:
                # print("Epsilon problem encountered for botcoins: {}".format(BOTCOINS))
                BOTCOINS = Decimal(0)

        CURRENT_ASSETS = BOTCOINS * BOTCOIN_PRICE
        CURRENT_ASSETS += CASH

        VALUE_HISTORY.append(CURRENT_ASSETS)
        LEVERAGE_HISTORY.append(soughtLeverage)
        DATETIME_HISTORY.append(now)

        # Print the state START

        # print("Today is {}".format(now))
        # print("Short term data range: {} - {}".format(currentShortTerm.date, currentShortTerm.endDate))
        # print("Long term data range: {} - {}".format(currentLongTerm.date, currentLongTerm.endDate))
        # print("Sought leverage: {}".format(soughtLeverage))
        # print("Current Value: {} ({} botcoins @ {} + {} cash)".format(CURRENT_ASSETS, BOTCOINS, BOTCOIN_PRICE, CASH))
        # print("")

        # Print the state END
        now += timeSteps


    # print(CURRENT_ASSETS)
    return {
        "success" : ((CURRENT_ASSETS - startingCash) / startingCash) * 100,
        "valueHistory" : VALUE_HISTORY,
        "leverageHistory" : LEVERAGE_HISTORY,
        "chartingParameters" : CHARTING_PARAMETERS_HISTORY,
        "dateTimeHistory" : DATETIME_HISTORY,
    }
       
def simulationPlotter(longTermData, valueHistory, leverageHistory, chartingParameters, dateHistory):
    assert isinstance(longTermData, list)
    assert isinstance(valueHistory, list)
    assert isinstance(leverageHistory, list)
    assert isinstance(chartingParameters, dict)

    assert len(valueHistory) == len(leverageHistory)
    assert len(leverageHistory) == len(dateHistory)
    for key, value in chartingParameters.items():
        assert len(value) == len(valueHistory), "{} has {} items, when it should have {}".format(key, len(value), len(valueHistory))


    preMainFrame = []
    preValueFrame = []
    preLeverageFrame = []
    preOtherFrames = []
    currentDatePointer = 0
    for x in longTermData:
        # print("")
        # print("LONG TERM:", x.date)
        valueData = {
            "Date" : x.date,
            "value" : [],
        }
        leverageData = {
            "Date" : x.date,
            "leverage" : [],
        }
        relevantData = {
            "Date" : x.date,
        }
        for key in chartingParameters.keys():
            relevantData[key] = []
        while dateHistory[currentDatePointer] < x.date:
            # print("Skip", dateHistory[currentDatePointer])
            currentDatePointer += 1
        while dateHistory[currentDatePointer] < x.endDate:
            # print("Accept", dateHistory[currentDatePointer])
            valueData["value"].append(valueHistory[currentDatePointer])
            leverageData["leverage"].append(leverageHistory[currentDatePointer])
            currentDatePointer += 1
            if currentDatePointer >= len(dateHistory):
                break
            for key, value in chartingParameters.items():
                if value[currentDatePointer] != None:
                    relevantData[key].append(value[currentDatePointer])


        for key, value in relevantData.items():
            if key not in ["Date"]:
                if len(value) == 0:
                    relevantData[key] = None
                else:
                    relevantData[key] = mean(value)
        for key, value in valueData.items():
            if key not in ["Date"]:
                if len(value) == 0:
                    valueData[key] = 0
                else:
                    valueData[key] = mean(value)
        for key, value in leverageData.items():
            if key not in ["Date"]:
                if len(value) == 0:
                    leverageData[key] = 0
                else:
                    leverageData[key] = mean(value)

        preOtherFrames.append(relevantData)
        preValueFrame.append(valueData)
        preLeverageFrame.append(leverageData)

        currentMainFrame = {
            "Date" : x.date,
            "Open" : x.open,
            "Close" : x.close,
            "High" : x.high,
            "Low" : x.low,
            "Volume" : x.volume,
        }
        preMainFrame.append(currentMainFrame)

    mainFrame = pd.DataFrame(preMainFrame)
    mainFrame.set_index("Date", inplace=True)

    otherFrame = pd.DataFrame(preOtherFrames)
    otherFrame.set_index("Date", inplace=True)

    leverageFrame = pd.DataFrame(preLeverageFrame)
    leverageFrame.set_index("Date", inplace=True)
    
    valueFrame = pd.DataFrame(preValueFrame)
    valueFrame.set_index("Date", inplace=True)


    addplots = [
        mpf.make_addplot(otherFrame, type="line", panel=0),
        mpf.make_addplot(leverageFrame, type="scatter", panel=2, ylabel="Leverage", color="orange"),
        mpf.make_addplot(valueFrame, type="line", panel=2, ylabel="Value"),
    ]
    mpf.plot(mainFrame, type="candle", volume=True, addplot=addplots, main_panel=0, volume_panel=1, num_panels=3)


def getData(filename, startDate, endDate, shortTermWindow, longTermWindow):
    assert isinstance(filename, str)
    assert isinstance(startDate, datetime)
    assert isinstance(endDate, datetime)
    assert isinstance(shortTermWindow, timedelta)
    assert isinstance(longTermWindow, timedelta)
    uniqueHash = "{}_{}_{}_{}_{}.pickle".format(filename, startDate.timestamp(), endDate.timestamp(), shortTermWindow.total_seconds(), longTermWindow.total_seconds())

    CACHED_DATA_FOLDER = "cache"
    cacheDataFolderPath = os.path.join(pathlib.Path().resolve(), CACHED_DATA_FOLDER)
    if not os.path.isdir(cacheDataFolderPath):
        os.mkdir(cacheDataFolderPath)

    filePath = os.path.join(cacheDataFolderPath, uniqueHash)

    if os.path.isfile(filePath):
        file = open(filePath, "rb")
        print("{} loaded from cache".format(uniqueHash))
        data = pickle.load(file)
        file.close()
        return data
    
    botcoin = RawData(filename)
    dateRange = endDate-startDate
    allData = botcoin.fetchData(startDate, dateRange)
    print("All data between {} and {} has been fetched.".format(startDate, endDate))


    convertDataThread = ConvertDataMultiProcess(allData, startDate, dateRange)
    p = multiprocessing.Pool(2)
    shortTerm, longTerm = p.map(convertDataThread.convertData, [shortTermWindow, longTermWindow])

    data = {
        "short" : shortTerm,
        "long" : longTerm,
    }
    file = open(filePath, "wb")
    pickle.dump(data, file)
    file.close()
    return data

class HypothesisTester:
    def __init__(self, startingDate, shortTermWindow, endingDate, shortTermData, longTermData, startingCash):
        self.startingDate = startingDate
        self.shortTermWindow = shortTermWindow
        self.endingDate = endingDate
        self.shortTermData = shortTermData
        self.longTermData = longTermData
        self.startingCash = startingCash

    def testHypothesis(self, hypothesis):
        assert callable(hypothesis)
        return simulation(self.startingDate, self.shortTermWindow, self.endingDate, self.shortTermData, self.longTermData, hypothesis, self.startingCash)["success"]

def DataFrame(data):
    
    short = pd.DataFrame([val.__dict__ for val in data['short']]).dropna()
    long = pd.DataFrame([val.__dict__ for val in data['long']]).dropna()
    #plt.plot_date(short.date,short.safeMeanPrice, linestyle='solid', marker='')
        
    #check for linearity with scatter plots
    
    #plt.scatter(short.safeMeanDeltaVolumePerTransaction, short.safeMeanPrice)
    
    #create additional data
    def signMomentum(sign):
        momentum = []
        mom = 0
        LastSignPositive = True
        for index, sign in enumerate(sign):
            if sign == 0: #concern, we should probably count very small differences as 0 instead of increasing momentum
                mom = 0
            if sign == 1:
                if LastSignPositive == True:
                    mom += 1
                else:
                    mom = 1
                LastSignPositive = True
            if sign == -1:
                if LastSignPositive == True:
                    mom = -1
                else:
                    mom -= 1
                LastSignPositive = False
            if np.isnan(sign):
                momentum.append(np.nan)
            else:
                momentum.append(mom)
        return momentum
    short['deltaPrice1Row'] = short['safeMeanPrice'].diff()
    short['deltaPrice5Row'] = short['safeMeanPrice'].diff(periods=5)
    short['deltaPrice10Row'] = short['safeMeanPrice'].diff(periods=10)
    short['deltaPrice25Row'] = short['safeMeanPrice'].diff(periods=25)
    short['deltaPrice50Row'] = short['safeMeanPrice'].diff(periods=50)
    short['deltaPrice100Row'] = short['safeMeanPrice'].diff(periods=100)
    short['deltaPrice200Row'] = short['safeMeanPrice'].diff(periods=200)
    short['deltaPrice500Row'] = short['safeMeanPrice'].diff(periods=500)
    #^^ if you graph all of these with the above scatter, you will find linearity starting to increase at >100 rows ^^ which makes some sense
    
    
    short['deltaSign1Row'] = np.sign(short['deltaPrice1Row'])
    short['signMomentum1Row'] = signMomentum(short['deltaSign1Row'])
    short['deltaSign500Row'] = np.sign(short['deltaPrice500Row'])
    short['signMomentum500Row'] = signMomentum(short['deltaSign500Row'])
    short['std5Row'] = short['safeMeanPrice'].rolling(5).std()
    short['std100Row'] = short['safeMeanPrice'].rolling(100).std()
    short['volume5Row'] = short['volume'].rolling(5).sum()
    short['volume100Row'] = short['volume'].rolling(100).sum()
    short['volume500Row'] = short['volume'].rolling(500).sum()
    
    short['movingAverage5'] = short['safeMeanPrice'].rolling(5).sum()/5
    short['movingAverage50'] = short['safeMeanPrice'].rolling(50).sum()/50
    short['movingAverage500'] = short['safeMeanPrice'].rolling(500).sum()/500
    
    plt.scatter(short.deltaPrice500Row, short.safeMeanPrice)
    plt.scatter(short.signMomentum1Row, short.safeMeanPrice)
    
    
    #linear-ish combos: std5row, safemeanprice
    #signmomentum500row, deltaprice500row -- also on log scale
    #signmomentum500row * volume500Row, deltaprice500row
    
    
    
    with pd.ExcelWriter('discretedata.xlsx') as writer:  
        short.to_excel(writer, sheet_name='short')
        long.to_excel(writer, sheet_name='long')


    return short,long



def main():
    THREAD_COUNT = os.cpu_count()
    print("I have {} cores".format(THREAD_COUNT))
    FILENAME = "XMRUSD.csv"

    startingDate = datetime(year=2017, month=1, day=1, hour=0, minute=0, second=0)
    endingDate = datetime(year=2017, month=8, day=1)
    shortTermWindow = timedelta(hours=1)
    longTermWindow = timedelta(hours=24)

    import time
    start = time.time()
    data = getData(FILENAME, startingDate, endingDate, shortTermWindow, longTermWindow)
    shortdf,longdf = DataFrame(data)
    print("Took {} seconds".format(time.time() - start))
    # for x in longTerm:
    #     print("L RANGE:", x.date, " - ", x.endDate)
    # # for x in shortTerm:
    # #     print("S RANGE:", x.date, " - ", x.endDate)

    result = simulation(startingDate, shortTermWindow, endingDate, data["short"], data["long"], hypothesis.bollingerBandsSafe, Decimal(1_000))
    print("{}% success".format(result["success"]))    
    simulationPlotter(data["long"], result["valueHistory"], result["leverageHistory"], result["chartingParameters"], result["dateTimeHistory"])


    # hypothesisTester = HypothesisTester(startingDate, shortTermWindow, endingDate, data["short"], data["long"], Decimal(1_000)).testHypothesis

    # inputList = np.arange(.01, 1, .01)
    # hypothesisList = [hypothesis.HypothesisVariation(hypothesis.bollingerBandsSafe, bollinger_number_of_stdev=i).hypothesis for i in inputList]

    # pool = multiprocessing.Pool(THREAD_COUNT)
    # results = pool.map(hypothesisTester, hypothesisList)
    # associatedDict = {}
    # for i in range(len(results)):
    #     associatedDict[inputList[i]] = results[i]

    #     print("Stdev {} : {}% profit".format(inputList[i], results[i]))
    
    # bestResult = max(results)
    # bestResultIndex = results.index(bestResult)
    # print("Best: {} : {}% profit".format(inputList[bestResultIndex], results[bestResultIndex]))

    
if __name__ == "__main__":
    main()
    

