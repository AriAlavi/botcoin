import csv
from datetime import datetime, timedelta

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
        print("end dateint:", endDateInt)
        
        for transaction in self.data:
            if  givenDate <= datetime.fromtimestamp(transaction.time) <= endDate:
                FETCHED_DATA.append(transaction)
            if transaction.time > endDateInt:
                break

        return FETCHED_DATA

class DiscreteData:
    def __init__(self, rawData):
        assert isinstance(rawData, list)
        assert all(isinstance(x, DataPoint) for x in rawData)

        self.price = None
        self.deltaPrice = None
        self.deltaDeltaPrice = None

        self.volume = None
        self.deltaVolume = None
        self.deltaDeltaVolume = None

        self.stdev = None
        self.deltaStdev = None
        self.deltaDeltaStdev = None

        self.volumePerTransaction = None
        self.minPrice = None
        self.maxPrice = None

        self.starTime = None
        self.endTime = None

        self.transactions = None





def main():
    filename = "XMRUSD.csv"
    botcoin = BotCoin(filename)
    randomDate = datetime(year=2017, month=4, day=20, hour=6, minute=9, second=6)
    print(botcoin.fetchData(randomDate, timedelta(hours=3)))
    print(botcoin.fetchData(randomDate, timedelta(hours=1)))
    print(botcoin.fetchData(randomDate, timedelta(minutes=1)))
    print(botcoin.fetchData(randomDate, timedelta(days=1)))
    
    
    

    
if __name__ == "__main__":
    main()
    

