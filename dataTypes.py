from datetime import datetime, timedelta
from statistics import median, mean, stdev

from numpy import isin

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


class DiscreteData:
    def __init__(self, rawData, startDate, timestep):
        assert isinstance(rawData, list)
        assert all(isinstance(x, DataPoint) for x in rawData)
        assert isinstance(startDate, datetime)
        assert isinstance(timestep, timedelta)

        safeMean = DiscreteData.safeMean

        self.date = startDate
        self.endDate = self.date + timestep

        self.safeMeanPrice = safeMean([x.price for x in rawData])
        self.meanPrice = self.safeMeanPrice
        self.volume = sum(x.quantity for x in rawData)

        if len(rawData) < 1:
            self.low = None
            self.high = None
            self.open = None
            self.close = None
            self.price = None

        else:
            self.low = min(x.price for x in rawData)
            self.high = max(x.price for x in rawData)
            self.open= rawData[0].price
            self.close = rawData[-1].price
            self.price = (self.low + self.high + self.open + self.close)/4


        self.transactions = len(rawData)

    def __str__(self):
        return str(self.date)

    def __repr__(self):
        return str(self)

    @staticmethod
    def safeMean(input):
        assert isinstance(input, list)
        if len(input) == 0:
            return None
        elif len(input) == 1:
            return input[0]
        else:
            return mean(input)




def convertData(rawData,givenDate,dateRange,givenWindow):
    assert isinstance(rawData, list)
    assert all(isinstance(x, DataPoint) for x in rawData)
    assert isinstance(givenDate, datetime)
    assert isinstance(dateRange,timedelta)
    assert isinstance(givenWindow,timedelta)
    # print(givenDate, dateRange)
    endDate = givenDate + dateRange

    currentDate = givenDate
    createdDiscreteData = []
    rawDataIndex = 0
    startDateInt = int(givenDate.timestamp())
    while currentDate < endDate:
        currentDateInt = int(currentDate.timestamp())
        currentWindowRawData = []
        currentData = rawData[rawDataIndex]
        while currentData.time <= currentDateInt:
            # print(currentData.time, startDateInt, currentData.time > startDateInt)
            # if currentData.time >= startDateInt:
            currentWindowRawData.append(currentData)

            rawDataIndex += 1
            currentData = rawData[rawDataIndex]

        createdObj = DiscreteData(currentWindowRawData, currentDate, givenWindow)
        createdDiscreteData.append(createdObj)
        currentDate += givenWindow
        

    return createdDiscreteData


# def convertData(rawData,givenDate,dateRange,givenWindow):
#     assert isinstance(rawData, list)
#     assert all(isinstance(x, DataPoint) for x in rawData)
#     assert isinstance(givenDate, datetime)
#     assert isinstance(dateRange,timedelta)
#     assert isinstance(givenWindow,timedelta)
    
#     rawDataList = []
#     datetimeList = []
#     beginDate = givenDate
#     endDate = givenDate + givenWindow
#     endDateInt = int(endDate.timestamp())
#     datetimeList.append(givenDate)
    
#     while givenDate < beginDate + dateRange:
#         sampleData = []
#         for transaction in rawData:
#             if  givenDate <= datetime.fromtimestamp(transaction.time) <= endDate:
#                 sampleData.append(transaction)
#             if transaction.time > endDateInt:
#                 continue
        
#         rawData = list(set(rawData) - set(sampleData))
#         rawDataList.append(sampleData)
#         givenDate = endDate
#         datetimeList.append(givenDate)
#         endDate = endDate + givenWindow
    

#     map_object = map(DiscreteData, rawDataList, datetimeList, [givenWindow] * len(rawDataList))
#     newDataList = list(map_object)
#     return newDataList

def splitDate(startDate, dateRange, splitPieces):
    assert isinstance(startDate, datetime)
    assert isinstance(dateRange, timedelta)
    assert isinstance(splitPieces, int)
    splitRange = dateRange / splitPieces

    END_DATE = startDate + dateRange

    HOUR = 3_600

    if not splitRange.total_seconds() % HOUR == 0:
        totalSeconds = splitRange.total_seconds()
        remainderSeconds = splitRange.total_seconds() % HOUR
        # print("TOTAL", totalSeconds)
        # print("REMAINDER", remainderSeconds)
        # print("ADDING", HOUR)
        splitRange = timedelta(seconds=(totalSeconds - remainderSeconds) + HOUR)

    assert (splitRange.seconds % 3600) == 0, "was {} instead".format(splitRange.seconds)
    datePieces = []
    currentDate = startDate
    while currentDate < END_DATE:
        datePiece = [currentDate, splitRange]
        datePieces.append(datePiece)
        currentDate += splitRange
    
    last = datePieces[-1]
    last[1] = END_DATE - last[0]
    datePieces[-1] = last
    return datePieces


def convertDataMultiProcess(allData, startDate, dateRange, windowLength, coreCount=20):
    result = convertData(allData, startDate, dateRange, windowLength)
    # jobs = splitDate(startDate, dateRange, coreCount)
    # results = []
    # for date, length in jobs:
    #     result = convertData(allData, date, length, windowLength)
    #     results.extend(result)
    #     # print(date, length, result)


    return result