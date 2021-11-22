from datetime import datetime, timedelta
from statistics import median, mean, stdev

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
        getDelta = DiscreteData.getDelta

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

    @staticmethod
    def safeMean(input):
        assert isinstance(input, list)
        if len(input) == 0:
            return None
        elif len(input) == 1:
            return input[0]
        else:
            return mean(input)

    @staticmethod
    def getDelta(givenList):
        assert isinstance(givenList, list)
        deltaList = []
        if len(givenList) <= 1:
            return deltaList
        for i in range(1, len(givenList)):
            delta = givenList[i] - givenList[i-1]
            deltaList.append(delta)
        return deltaList