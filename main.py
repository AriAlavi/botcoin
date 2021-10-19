import csv
from datetime import datetime, timedelta

FILENAME = "XMRUSD.csv"

def readUnix(givenTimestamp) -> str:
    assert isinstance(givenTimestamp, int)
    return str(datetime.fromtimestamp(givenTimestamp).strftime("%m/%d/%Y %H:%M"))

def timestampToDatetime(givenTiemstamp) -> datetime:
    assert isinstance(givenTiemstamp, int)
    return datetime.fromtimestamp(givenTiemstamp)

class DataPoint:
    def __init__(self, time, price, quantity):
        assert isinstance(time, int)
        assert isinstance(quantity, float)
        assert isinstance(price, float)

        self.time = time
        self.quantity = quantity
        self.price = price

    def __str__(self):
        return "[{}] {} @ ${}".format(readUnix(self.time), self.quantity, self.price)

    def __repr__(self):
        return str(self)


def readFile(filename):
    assert isinstance(filename, str)
    file = open(filename)
    reader = csv.reader(file)
    i = 0
    DATA_COLLECTION = []
    for row in reader:
        DATA_COLLECTION.append(DataPoint(int(row[0]), float(row[1]), float(row[2])))

    file.close()
    print("File {} loaded!".format(filename))
    return DATA_COLLECTION
    
def fetchData(data, givenDate, givenWindow):
    assert isinstance(data, list)
    assert isinstance(givenDate, datetime)
    assert isinstance(givenWindow, timedelta)
    FETCHED_DATA = []
    
    endDate = givenDate.total_seconds() + givenWindow.total_seconds()
    
    for transaction in data:
        if  givenDate <= transaction.time <= endDate:
            FETCHED_DATA.append(transaction)
        if transaction.time == endDate:
            break

    return FETCHED_DATA

def main():
    data = readFile(FILENAME)
    randomDate = datetime(year=2020, month=4, day=20, hour=5, minute=10, second=35)

    print(fetchData(data, randomDate, timedelta(hours=3)))
    print(fetchData(data, randomDate, timedelta(hours=1)))
    print(fetchData(data, randomDate, timedelta(minutes=1)))
    print(fetchData(data, randomDate, timedelta(days=1)))
    
if __name__ == "__main__":
    main()
