import csv
from datetime import date, datetime, timedelta
from multiprocessing import Pool
from decimal import Decimal
import multiprocessing
import pickle
import pathlib
import os

from dataTypes import *
from simulation import *
import hypothesis




class HypothesisTesterStupid:
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

def DataFrame(data, filename):
    assert isinstance(filename, str)
    assert ".xlsx" in filename
    
    short = pd.DataFrame([val.__dict__ for val in data['short']]).dropna()
    long = pd.DataFrame([val.__dict__ for val in data['long']]).dropna()
    #plt.plot_date(short.date,short.safeMeanPrice, linestyle='solid', marker='')
        
    #check for linearity with scatter plots
    
    #plt.scatter(short.safeMeanDeltaVolumePerTransaction, short.safeMeanPrice)
    
    #create additional data
    # def signMomentum(sign):
    #     momentum = []
    #     mom = 0
    #     LastSignPositive = True
    #     for index, sign in enumerate(sign):
    #         if sign == 0: #concern, we should probably count very small differences as 0 instead of increasing momentum
    #             mom = 0
    #         if sign == 1:
    #             if LastSignPositive == True:
    #                 mom += 1
    #             else:
    #                 mom = 1
    #             LastSignPositive = True
    #         if sign == -1:
    #             if LastSignPositive == True:
    #                 mom = -1
    #             else:
    #                 mom -= 1
    #             LastSignPositive = False
    #         if np.isnan(sign):
    #             momentum.append(np.nan)
    #         else:
    #             momentum.append(mom)
    #     return momentum
    # short['deltaPrice1Row'] = short['safeMeanPrice'].diff()
    # short['deltaPrice5Row'] = short['safeMeanPrice'].diff(periods=5)
    # short['deltaPrice10Row'] = short['safeMeanPrice'].diff(periods=10)
    # short['deltaPrice25Row'] = short['safeMeanPrice'].diff(periods=25)
    # short['deltaPrice50Row'] = short['safeMeanPrice'].diff(periods=50)
    # short['deltaPrice100Row'] = short['safeMeanPrice'].diff(periods=100)
    # short['deltaPrice200Row'] = short['safeMeanPrice'].diff(periods=200)
    # short['deltaPrice500Row'] = short['safeMeanPrice'].diff(periods=500)
    # #^^ if you graph all of these with the above scatter, you will find linearity starting to increase at >100 rows ^^ which makes some sense
    
    
    # short['deltaSign1Row'] = np.sign(short['deltaPrice1Row'])
    # short['signMomentum1Row'] = signMomentum(short['deltaSign1Row'])
    # short['deltaSign500Row'] = np.sign(short['deltaPrice500Row'])
    # short['signMomentum500Row'] = signMomentum(short['deltaSign500Row'])
    # short['std5Row'] = short['safeMeanPrice'].rolling(5).std()
    # short['std100Row'] = short['safeMeanPrice'].rolling(100).std()
    # short['volume5Row'] = short['volume'].rolling(5).sum()
    # short['volume100Row'] = short['volume'].rolling(100).sum()
    # short['volume500Row'] = short['volume'].rolling(500).sum()
    
    # short['movingAverage5'] = short['safeMeanPrice'].rolling(5).sum()/5
    # short['movingAverage50'] = short['safeMeanPrice'].rolling(50).sum()/50
    # short['movingAverage500'] = short['safeMeanPrice'].rolling(500).sum()/500
    
    # plt.scatter(short.deltaPrice500Row, short.safeMeanPrice)
    # plt.scatter(short.signMomentum1Row, short.safeMeanPrice)
    
    
    #linear-ish combos: std5row, safemeanprice
    #signmomentum500row, deltaprice500row -- also on log scale
    #signmomentum500row * volume500Row, deltaprice500row
    
    
    
    with pd.ExcelWriter(filename) as writer:  
        short.to_excel(writer, sheet_name='short')
        long.to_excel(writer, sheet_name='long')


    return short,long



def convertDataListMode(convertMe):
    assert isinstance(convertMe, list)

    startingDate = datetime(year=2018, month=1, day=2, hour=0, minute=0, second=0)
    endingDate = datetime(year=2020, month=12, day=31)
    shortTermWindow = timedelta(hours=1)
    longTermWindow = timedelta(hours=24)
    for filename in convertMe:
        try:
            file = open(filename) # Check to see if exists and openable
            file.close()
        except:
            raise Exception(f"{convertMe} file not found!")
    for filename in convertMe:
        data = getData(filename, startingDate, endingDate, shortTermWindow, longTermWindow)
        excelName = filename.replace(".csv", ".xlsx")
        DataFrame(data, excelName)

def main():
    THREAD_COUNT = os.cpu_count()
    print("I have {} cores".format(THREAD_COUNT))
    FILENAME = "XMRUSD.csv"
    # print(hypothesisTester(FILENAME, hypothesis.equationMethod))
    

    startingDate = datetime(year=2018, month=1, day=2, hour=0, minute=0, second=0)
    endingDate = datetime(year=2020, month=12, day=31)
    shortTermWindow = timedelta(hours=1)
    longTermWindow = timedelta(hours=24)

    # import time
    # start = time.time()
    data = getData(FILENAME, startingDate, endingDate, shortTermWindow, longTermWindow)
    shortdf,longdf = DataFrame(data)
    # print("Took {} seconds".format(time.time() - start))
    # for x in longTerm:
    #     print("L RANGE:", x.date, " - ", x.endDate)
    # # for x in shortTerm:
    # #     print("S RANGE:", x.date, " - ", x.endDate)

    result = simulation(startingDate, shortTermWindow, endingDate, data["short"], data["long"], hypothesis.hold, Decimal(1_000))
   
    def ParameterPrint():
        return "{:.3f}% success\n{:.3f}% market risk\n{} Buys\n{} Sells"\
            .format(result["success"],result["marketRisk"],result["numberOfBuys"],result["numberOfSells"])

    print(ParameterPrint())
    

       

    
    # simulationPlotter(data["long"], result)
    # print(hypothesisTester(FILENAME, hypothesis.hold))

    # hypothesisTester = HypothesisTester(startingDate, shortTermWindow, endingDate, data["short"], data["long"], Decimal(1_000)).testHypothesis

    # inputList = np.arange(.05, 3, .05)
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
    
