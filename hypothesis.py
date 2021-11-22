from numpy.lib.arraysetops import isin
from dataTypes import *
from decimal import Decimal

def randomChoice(shortTerm, longTerm, cash, botcoins, customParameters, chartingParameters):
    assert isinstance(shortTerm, DiscreteData)
    assert isinstance(longTerm, DiscreteData)
    assert isinstance(cash, Decimal)
    assert isinstance(botcoins, Decimal)
    assert isinstance(customParameters, dict)
    assert isinstance(chartingParameters, dict)
    import random
    return Decimal(random.randint(0, 100))/100

def bounce(shortTerm, longTerm, cash, botcoins, customParameters, chartingParameters):
    assert isinstance(shortTerm, DiscreteData)
    assert isinstance(longTerm, DiscreteData)
    assert isinstance(cash, Decimal)
    assert isinstance(botcoins, Decimal), "{} instead".format(type(botcoins))
    assert isinstance(customParameters, dict)
    assert isinstance(chartingParameters, dict)

    sellAll = customParameters.get("sell", False)
    if sellAll:
        customParameters["sell"] = False
        return Decimal(0)
    else:
        customParameters["sell"] = True
        return Decimal(1)

def hold(*args):
    return Decimal(1)

def bollingerBandsSafe(shortTerm, longTerm, cash, botcoins, customParameters, chartingParameters):
    assert isinstance(shortTerm, DiscreteData)
    assert isinstance(longTerm, DiscreteData)
    assert isinstance(cash, Decimal)
    assert isinstance(botcoins, Decimal), "{} instead".format(type(botcoins))
    assert isinstance(customParameters, dict)
    assert isinstance(chartingParameters, dict)

    BOLLINGER_BAND_TIME_PERIOD = 20
    BOLLINGER_NUMBER_OF_STDEV = 1.5

    if len(customParameters.keys()) == 0:
        customParameters["history"] = []
        customParameters["lastLeverage"] = Decimal(0)
        customParameters["buySignal"] = False
        customParameters["sellSignal"] = False
        chartingParameters["lowerBound"] = []
        chartingParameters["upperBound"] = []

    lastLeverage = customParameters["lastLeverage"]
    
    if len(customParameters["history"]) == 0:
        customParameters["history"].append(longTerm)
    else:
        if longTerm.date != customParameters["history"][-1].date:
            customParameters["history"].append(longTerm)
        if len(customParameters["history"]) > BOLLINGER_BAND_TIME_PERIOD:
            customParameters["history"].pop(0)

    history = customParameters["history"]
    if len(history) <= BOLLINGER_BAND_TIME_PERIOD / 10:
        chartingParameters["lowerBound"] = None
        chartingParameters["upperBound"] = None
        return Decimal(0)

    currentPrice = shortTerm.safeMeanPrice

    movingAveragePrice = mean([x.safeMeanPrice for x in history])
    movingAverageStdev = stdev([x.safeMeanPrice for x in history])
    upperBound = movingAveragePrice + (movingAverageStdev * BOLLINGER_NUMBER_OF_STDEV)
    lowerBound = movingAveragePrice - (movingAverageStdev * BOLLINGER_NUMBER_OF_STDEV)

    chartingParameters["lowerBound"] = lowerBound
    chartingParameters["upperBound"] = upperBound

    if not currentPrice:
        return lastLeverage

    # print("AVG PRICE", movingAveragePrice, "STDEV", movingAverageStdev, "PRICE NOW", shortTerm.safeMeanPrice)

    if currentPrice > upperBound:
        customParameters["sellSignal"] = True
        customParameters["buySignal"] = False
    elif currentPrice < lowerBound:
        customParameters["sellSignal"] = False
        customParameters["buySignal"] = True
    else:
        if customParameters["buySignal"]:
            customParameters["lastLeverage"] = Decimal(1)
            return Decimal(1)
        elif customParameters["sellSignal"]:
            customParameters["lastLeverage"] = Decimal(0)
            return Decimal(0)
   
    return lastLeverage