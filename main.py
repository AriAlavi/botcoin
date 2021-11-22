import csv
from datetime import datetime, timedelta
from statistics import median, mean, stdev
from multiprocessing import Pool
from decimal import Decimal
import os

class Precise():
    def __init__(self, value):
        self.value = value


class DataPoint:
    def __init__(self, time, price, quantity):
        assert isinstance(time, int)
        assert isinstance(quantity, float)
        assert isinstance(price, float)

        self.time = time
        self.quantity = quantity
        self.price = price

    def readUnix(self,givenTimestamp) -> str:
        assert isinstance(givenTimestamp, int)
        return str(datetime.fromtimestamp(givenTimestamp).strftime("%m/%d/%Y %H:%M"))

    def timestampToDatetime(self,givenTiemstamp) -> datetime:
        assert isinstance(givenTiemstamp, int)
        return datetime.fromtimestamp(givenTiemstamp)

    def __str__(self):
        return "[{}] {} @ ${}".format(self.readUnix(self.time), self.quantity, self.price)

    def __repr__(self):
        return str(self)


class BotCoin:
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
        print("File {} loaded!".format(self.filename))
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

def getDelta(givenList):
    assert isinstance(givenList, list)
    deltaList = []
    if len(givenList) <= 1:
        return deltaList
    for i in range(1, len(givenList)):
        delta = givenList[i] - givenList[i-1]
        deltaList.append(delta)
    return deltaList

def safeMean(input):
    assert isinstance(input, list)
    if len(input) == 0:
        return None
    elif len(input) == 1:
        return input[0]
    else:
        return mean(input)


class DiscreteData:
    def __init__(self, rawData, startDate, timestep):
        assert isinstance(rawData, list)
        assert all(isinstance(x, DataPoint) for x in rawData)
        assert isinstance(startDate, datetime)
        assert isinstance(timestep, timedelta)

        self.date = startDate
        self.endDate = self.date + timestep

        self.safeMeanPrice = safeMean([x.price for x in rawData])
        self.safeMeanDeltaPrice = safeMean(getDelta([x.price for x in rawData]))
        self.safeMeanDeltaDeltaPrice = safeMean(getDelta(getDelta([x.price for x in rawData])))

        self.volume = sum(x.quantity for x in rawData)

        self.safeMeanVolumePerTransaction = safeMean([x.quantity for x in rawData])
        self.safeMeanDeltaVolumePerTransaction = safeMean(getDelta([x.quantity for x in rawData]))
        self.safeMeanDeltaDeltaVolumePerTransaction = safeMean(getDelta(getDelta([x.quantity for x in rawData])))

        if len(rawData) < 2:
            self.priceStdev = None
            self.volumeStdev = None
        else:
            self.priceStdev = stdev(x.price for x in rawData)
            self.volumeStdev = stdev(x.quantity for x in rawData)

        if len(rawData) < 1:
            self.minPrice = None
            self.maxPrice = None

        else:
            self.minPrice = min(x.price for x in rawData)
            self.maxPrice = max(x.price for x in rawData)


        self.transactions = len(rawData)

def convertData(rawData,givenDate,dateRange,givenWindow):
    assert isinstance(rawData, list)
    assert all(isinstance(x, DataPoint) for x in rawData)
    assert isinstance(givenDate, datetime)
    assert isinstance(dateRange,timedelta)
    assert isinstance(givenWindow,timedelta)
    
    rawDataList = []
    datetimeList = []
    beginDate = givenDate
    endDate = givenDate + givenWindow
    endDateInt = int(endDate.timestamp())
    datetimeList.append(givenDate)
    
    while givenDate < beginDate + dateRange:
        sampleData = []
        for transaction in rawData:
            if  givenDate <= datetime.fromtimestamp(transaction.time) <= endDate:
                sampleData.append(transaction)
            if transaction.time > endDateInt:
                continue
        
        rawData = list(set(rawData) - set(sampleData))
        rawDataList.append(sampleData)
        givenDate = endDate
        datetimeList.append(givenDate)
        endDate = endDate + givenWindow
    

    map_object = map(DiscreteData, rawDataList, datetimeList, [givenWindow] * len(rawDataList))
    newDataList = list(map_object)
    return newDataList


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
    BOTCOIN_PRICE = Decimal(-1)
    VALUE_HISTORY = []
    LEVERAGE_HISTORY = []

    now = startingDate
    

    MAX_SHORT_TERM_INDEX = len(shortTermData) - 1
    MAX_LONG_TERM_INDEX = len(longtermData) - 1
    shortTermIndex = -3
    longTermIndex = 0

    LONG_TERM_BEGINS = longtermData[longTermIndex].endDate
    while now <= LONG_TERM_BEGINS:
        now += timeSteps
        shortTermIndex += 1

    epsilon = Decimal(.0000000001)

    customParameters = {}
    while now <= endingDate:
        if shortTermIndex < MAX_SHORT_TERM_INDEX:
            shortTermIndex += 1
        else:
            print("It is {} and the index is {} and the date of that index is {}. This shouldn't be possible, but I'm going to pretend like nothing has gone wrong.".format(now, shortTermIndex, shortTermData[shortTermIndex].date))

        
        currentShortTerm = shortTermData[shortTermIndex] 
        currentLongTerm = longtermData[longTermIndex]

        # print("Now {} vs Short term {} - {}".format(now, currentShortTerm.date, currentShortTerm.endDate))
        assert now == currentShortTerm.endDate

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

        soughtLeverage = hypothesisFunc(currentShortTerm, currentLongTerm, customParameters)
        newLeverage = setLeverage(CASH, BOTCOINS, BOTCOIN_PRICE, soughtLeverage)
        if BOTCOIN_PRICE <= 0:
            now += timeSteps
            continue
        CASH = newLeverage["cash"]
        BOTCOINS = newLeverage["coins"]
        if CASH < 0:
            if CASH + epsilon < 0:
                raise Exception("Cash cannot go negative! It is {}".format(CASH))
            else:
                CASH = 0

        if BOTCOINS < 0:
            if BOTCOINS + epsilon < 0:
                raise Exception("Cash cannot go negative! It is {}".format(BOTCOINS))
            else:
                BOTCOINS = 0

        CURRENT_ASSETS = BOTCOINS * BOTCOIN_PRICE
        CURRENT_ASSETS += CASH

        VALUE_HISTORY.append(CURRENT_ASSETS)
        LEVERAGE_HISTORY.append(soughtLeverage)

        # Print the state START

        # print("Today is {}".format(now))
        # print("Short term data range: {} - {}".format(currentShortTerm.date, currentShortTerm.endDate))
        # print("Long term data range: {} - {}".format(currentLongTerm.date, currentLongTerm.endDate))
        # print("Sought leverage: {}".format(soughtLeverage))
        # print("Current Value: {} ({} botcoins @ {} + {} cash)".format(CURRENT_ASSETS, BOTCOINS, BOTCOIN_PRICE, CASH))
        # print("")

        # Print the state END
        now += timeSteps



    return {
        "success" : ((CURRENT_ASSETS - startingCash) / startingCash) * 100,
        "valueHistory" : VALUE_HISTORY,
        "leverageHistory" : LEVERAGE_HISTORY,
    }
    
    



def main():
    THREAD_COUNT = os.cpu_count()
    print("I have {} cores".format(THREAD_COUNT))
    filename = "XMRUSD.csv"
    botcoin = BotCoin(filename)

    startingDate = datetime(year=2017, month=4, day=20, hour=0, minute=0, second=0)
    endingDate = datetime(year=2017, month=4, day=24)
    dateRange = timedelta(minutes=1)

    dateRange = endingDate-startingDate
    allData = botcoin.fetchData(startingDate, dateRange)
    shortTerm = convertData(allData, startingDate, dateRange, timedelta(hours=1))
    longTerm = convertData(allData, startingDate, dateRange, timedelta(hours=24))
    for x in longTerm:
        print("L RANGE:", x.date, " - ", x.endDate)
    # for x in shortTerm:
    #     print("S RANGE:", x.date, " - ", x.endDate)
    print(dateRange)

    def hypothesis(*args):
        import random
        return Decimal(random.randint(0, 100))/100

    # result = simulation(startingDate, timedelta(hours=1), endingDate, shortTerm, longTerm, hypothesis, Decimal(1_000))
    # print("{}% success".format(result["success"]))

    #DiscreteData(botcoin.fetchData(randomDate, timedelta(hours=4)))
    #DiscreteData(botcoin.fetchData(randomDate, timedelta(days=1)))
    #DiscreteData(botcoin.fetchData(randomDate, timedelta(minutes=1)))
    #DiscreteData([DataPoint(1, float(1), float(1))])

    # print(botcoin.fetchData(randomDate, timedelta(hours=3)))
    # print(botcoin.fetchData(randomDate, timedelta(hours=1)))
    # print(botcoin.fetchData(randomDate, timedelta(minutes=1)))
    # print(botcoin.fetchData(randomDate, timedelta(days=1)))
    
    
    

    
if __name__ == "__main__":
    main()
    

