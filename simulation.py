from datetime import timedelta, datetime
from decimal import Decimal
from dataTypes import *


import os, pickle, pathlib, csv, multiprocessing

import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

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

    # if os.path.isfile(filePath):
    #     file = open(filePath, "rb")
    #     print("{} loaded from cache".format(uniqueHash))
    #     data = pickle.load(file)
    #     file.close()
    #     return data
    
    botcoin = RawData(filename)
    dateRange = endDate-startDate
    allData = botcoin.fetchData(startDate, dateRange)
    print("All data between {} and {} has been fetched.".format(startDate, endDate))
    shortTerm = convertDataMultiProcess(allData, startDate, dateRange, shortTermWindow)
    longTerm = convertDataMultiProcess(allData, startDate, dateRange, longTermWindow)

    print("GOT TOTAL", len(shortTerm))
    for x in shortTerm:
        print(x, x.transactions)

    data = {
        "short" : shortTerm,
        "long" : longTerm,
    }
    file = open(filePath, "wb")
    pickle.dump(data, file)
    file.close()
    return data

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
    BOTCOIN_HELD = 0            
    NUMBER_OF_BUYS = 0           
    NUMBER_OF_SEllS = 0  
    ITERATIONS = 0
    ITERATIONS_WITH_BOTCOINS = 0

    

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

        if BOTCOIN_HELD < BOTCOINS: 
            NUMBER_OF_BUYS += 1

        if BOTCOIN_HELD > BOTCOINS:
            NUMBER_OF_SEllS += 1

        if BOTCOIN_HELD > 0:
            ITERATIONS_WITH_BOTCOINS += 1


            
        ITERATIONS += 1

        CURRENT_ASSETS = BOTCOINS * BOTCOIN_PRICE
        CURRENT_ASSETS += CASH

        VALUE_HISTORY.append(CURRENT_ASSETS)
        LEVERAGE_HISTORY.append(soughtLeverage)
        DATETIME_HISTORY.append(now)
        BOTCOIN_HELD = BOTCOINS 



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
        "averageLeverage" : mean(LEVERAGE_HISTORY),
        "valueHistory" : VALUE_HISTORY,
        "leverageHistory" : LEVERAGE_HISTORY,
        "chartingParameters" : CHARTING_PARAMETERS_HISTORY,
        "dateTimeHistory" : DATETIME_HISTORY,
        "numberOfBuys" : NUMBER_OF_BUYS, 
        "numberOfSells" : NUMBER_OF_SEllS, 
        "marketRisk" : (ITERATIONS_WITH_BOTCOINS/ITERATIONS) * 100,
    }
       
def simulationPlotter(longTermData, simulationData):
    assert isinstance(simulationData, dict)
    valueHistory = simulationData["valueHistory"]
    leverageHistory = simulationData["leverageHistory"]
    chartingParameters = simulationData["chartingParameters"]
    dateHistory = simulationData["dateTimeHistory"]
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

def mapRunner(givenArg):
    return givenArg()

def hypothesisTester(FILENAME, hypothesis):
    THREAD_COUNT = os.cpu_count()
    print("I have {} cores".format(THREAD_COUNT))


    shortTermWindow = timedelta(hours=1)
    longTermWindow = timedelta(hours=24)

    DATERANGES = [
        (datetime( month=6, day=18, year=2020), datetime(month=1, day=11, year=2021)), # good
        (datetime( month=1, day=1, year=2017), datetime(month=6, day=21, year=2018)), #good 
        (datetime( month=12, day=1, year=2018), datetime(month=7, day=21, year=2019)), #good 
        (datetime( month=6, day=21, year=2018), datetime(month=11, day=11, year=2018)), #neutral
        (datetime( month=1, day=24, year=2020), datetime(month=7, day=6, year=2020)), #neutral
        (datetime( month=10, day=19, year=2017), datetime(month=8, day=15, year=2018)), #even
        (datetime( month=11, day=1, year=2018), datetime(month=4, day=20, year=2020)), #even
        (datetime( month=12, day=8, year=2017), datetime(month=12, day=30, year=2019)), #bad 
        (datetime( month=2, day=13, year=2021), datetime(month=7, day=25, year=2021)), #bad
        (datetime( month=8, day=26, year=2017), datetime(month=12, day=28, year=2018)), #bad
    ] 

    hypothesisList = []
    for x, y in DATERANGES:
        data = getData(FILENAME, x, y, shortTermWindow, longTermWindow)
        stupid = HypothesisTester(x, shortTermWindow, y, data["short"], data["long"], Decimal(1_000), hypothesis)
        hypothesisList.append(stupid)

    hypothesisList = [x.run for x in hypothesisList]
    p = multiprocessing.Pool(THREAD_COUNT)
    resultsList = list(p.map(mapRunner, hypothesisList))


    # for x in hypothesisList:
    #     resultsList.append(x(hypothesis))

    # htmap = map(hypothesisList, staticHypothesisList)
    # htlist = list(htmap)
    # print(htmap)
    # htlist = list(htmap)
    return mean(resultsList)
class HypothesisTester:
    def __init__ (self, startingDate, timeSteps, endingDate, shortTermData, longtermData, startingCash, hypothesis):
        self.startingDate = startingDate
        self.timeSteps = timeSteps
        self.endingDate = endingDate
        self.shortTermData = shortTermData
        self.longtermData = longtermData
        self.startingCash = startingCash
        self.hypothesis = hypothesis
    def run(self):
        return simulation(self.startingDate, self.timeSteps, self.endingDate, self.shortTermData, self.longtermData, self.hypothesis, self.startingCash).get('success')
    