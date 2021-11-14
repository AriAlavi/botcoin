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
    def __init__(self, rawData):
        assert isinstance(rawData, list)
        assert all(isinstance(x, DataPoint) for x in rawData)

    

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

            self.startTime = None
            self.endTime = None
        else:
            self.minPrice = min(x.price for x in rawData)
            self.maxPrice = max(x.price for x in rawData)

            self.startTime = min(x.time for x in rawData)
            self.endTime = max(x.time for x in rawData)

        self.transactions = len(rawData)

def convertData(rawData,givenDate,dateRange,givenWindow):
    assert isinstance(rawData, list)
    assert all(isinstance(x, DataPoint) for x in rawData)
    assert isinstance(givenDate, datetime)
    assert isinstance(dateRange,timedelta)
    assert isinstance(givenWindow,timedelta)
    
    rawDataList = []
    beginDate = givenDate
    endDate = givenDate + givenWindow
    endDateInt = int(endDate.timestamp())
    
    while endDate <= beginDate + dateRange:
        sampleData = []
        for transaction in rawData:
            if  givenDate <= datetime.fromtimestamp(transaction.time) <= endDate:
                sampleData.append(transaction)
            if transaction.time > endDateInt:
                continue
        
        rawData = list(set(rawData) - set(sampleData))
        rawDataList.append(sampleData)
        givenDate = endDate
        endDate = endDate + givenWindow
    

    map_object = map(DiscreteData, rawDataList)
    newDataList = list(map_object)
    return newDataList


def setLeverage(myCash, myCoins, coinPrice, soughtLeverage):
    assert isinstance(myCash, Decimal)
    assert isinstance(myCoins, Decimal)
    assert isinstance(coinPrice, Decimal)
    assert isinstance(soughtLeverage, Decimal)
    assert Decimal("0") <= soughtLeverage <= Decimal("1"), "was {} instead".format(float(soughtLeverage))
    assert myCash >= 0
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
        # soughtCoinValue = soughtLeverage * totalValue
        # toSpendOnCoins = soughtCoinValue - coinValue
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
    assert callable(hypothesisFunc)
    assert isinstance(startingCash, int) or isinstance(startingCash, float)
    
    assert startingDate < endingDate

    CASH = startingCash
    BOTCOINS = 0
    BOTCOIN_PRICE = 1

    now = startingDate
    while now <= endingDate:
        print("It is {}".format(now))

        soughtLeverage = hypothesisFunc(None, 4, "Ah")
        newLeverage = setLeverage(CASH, BOTCOINS, BOTCOIN_PRICE, soughtLeverage)
        CASH = newLeverage["cash"]
        BOTCOINS = newLeverage["coins"]



        now += timeSteps



def main():
    THREAD_COUNT = os.cpu_count()
    print("I have {} cores".format(THREAD_COUNT))
    filename = "XMRUSD.csv"
    botcoin = BotCoin(filename)

    startingDate = datetime(year=2017, month=4, day=20, hour=6, minute=9, second=6)
    endingDate = datetime(year=2017, month=5, day=1)
    dateRange = timedelta(minutes=1)

    dateRange = endingDate-startingDate
    allData = botcoin.fetchData(startingDate, dateRange)
    shortTerm = convertData(allData, startingDate, dateRange, timedelta(hours=1))
    longTerm = convertData(allData, startingDate, dateRange, timedelta(hours=24))

    def hypothesis(**args):
        import random
        return random.randint(0, 100)

    result = simulation(startingDate, timedelta(hours=1), endingDate, shortTerm, longTerm, hypothesis, 1_000)
    

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
    

