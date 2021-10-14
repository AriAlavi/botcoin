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

    # 1. ITERATE THROUGH ALL THE DATA
    # 2. COMPUTE THE END DATE (timedelta + datetime = datetime)
    # 3. CHECK IF THE ITEREATED DATE IS GREATER THAN OR EQUAL TO THE GIVEN DATE AND IF THE ITEREATED DATE IS LESS THAN OR EQUAL TO THE END DATE
    # 4. IF IT IS WITHIN THE REQUESTED RANGE, APPEND IT TO FETCHED DATA
    # (optimize) 5. IF THE PREVIOUS ITEM WAS APPENDED, BUT THE CURRENT WAS NOT APPENDED, BREAK FROM THE LOOP


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
