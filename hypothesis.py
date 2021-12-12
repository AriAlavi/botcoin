from dataTypes import *
from decimal import Decimal
from math import sin, cos, sqrt
from scipy.stats import linregress

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


def equationMethod(shortTerm, longTerm, cash, botcoins, customParameters, chartingParameters, **kwargs):
    assert isinstance(shortTerm, DiscreteData)
    assert isinstance(longTerm, DiscreteData)
    assert isinstance(cash, Decimal)
    assert isinstance(botcoins, Decimal), "{} instead".format(type(botcoins))
    assert isinstance(customParameters, dict)
    assert isinstance(chartingParameters, dict)

    def getDelta(givenList):
        assert isinstance(givenList, list)
        deltaList = []
        if len(givenList) <= 1:
            return deltaList
        for i in range(1, len(givenList)):
            delta = givenList[i] - givenList[i-1]
            deltaList.append(delta)
        return deltaList

    def linearEQ(givenDeltas):
        errorDelta = [abs(x-1) for x in givenDeltas]
        totalDelta = [abs(x) for x in givenDeltas]
        return sum(errorDelta)/sum(totalDelta)

    LONG_TERM_HISTORY_TIME_PERIOD = kwargs.get("long_term_history_time_period", 20)
    SHORT_TERM_HISTORY_TIME_PERIOD = kwargs.get("short_term_history_time_period", 20*24)
    MAP_EQUATIONS = {
        linearEQ : .75,
    }

    if len(customParameters.keys()) == 0:
        customParameters["longTermHistory"] = []
        customParameters["shortTermHistory"] = []
    if len(customParameters["longTermHistory"]) == 0:
        customParameters["longTermHistory"].append(longTerm)
    else:
        if longTerm.date != customParameters["longTermHistory"][-1].date:
            customParameters["longTermHistory"].append(longTerm)
        if len(customParameters["longTermHistory"]) > LONG_TERM_HISTORY_TIME_PERIOD:
            customParameters["longTermHistory"].pop(0)
    longTermHistory = customParameters["longTermHistory"]

    if len(customParameters["shortTermHistory"]) == 0:
        customParameters["shortTermHistory"].append(shortTerm)
    else:
        if longTerm.date != customParameters["shortTermHistory"][-1].date:
            customParameters["shortTermHistory"].append(shortTerm)
        if len(customParameters["shortTermHistory"]) > SHORT_TERM_HISTORY_TIME_PERIOD:
            customParameters["shortTermHistory"].pop(0)
    shortTermHistory = customParameters["shortTermHistory"]

    longTermPrices = []
    last = None
    for x in longTermHistory:
        if x.low:
            price = (x.low+x.high+x.open+x.close)/4
            last = price
            longTermPrices.append(price)
        elif last:
            longTermPrices.append(last)

    shortTermPrices = []
    last = None
    for x in shortTermHistory:
        if x.low:
            price = (x.low+x.high+x.open+x.close)/4
            last = price
            shortTermPrices.append(price)
        elif last:
            shortTermPrices.append(last)


    longTermDeltas = getDelta(longTermPrices)
    if len(longTermDeltas) == 0:
        chartingParameters["test"] = None
        return Decimal(0)

    slope, intercept, r_value, p_value, std_err = linregress(range(0, len(longTermPrices)), longTermPrices)
    shortTermDeltas = getDelta(shortTermPrices)
    chartingParameters["test"] = r_value**2
    return Decimal(r_value**2)


def bollingerBandsSafe(shortTerm, longTerm, cash, botcoins, customParameters, chartingParameters, **kwargs):
    assert isinstance(shortTerm, DiscreteData)
    assert isinstance(longTerm, DiscreteData)
    assert isinstance(cash, Decimal)
    assert isinstance(botcoins, Decimal), "{} instead".format(type(botcoins))
    assert isinstance(customParameters, dict)
    assert isinstance(chartingParameters, dict)

    BOLLINGER_BAND_TIME_PERIOD = kwargs.get("bollinger_band_time_period", 20)
    BOLLINGER_NUMBER_OF_STDEV = kwargs.get("bollinger_number_of_stdev", .1)

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

    try:
        movingAveragePrice = mean([x.safeMeanPrice for x in history if x.safeMeanPrice])
        movingAverageStdev = stdev([x.safeMeanPrice for x in history if x.safeMeanPrice])
    except:
        chartingParameters["lowerBound"] = None
        chartingParameters["upperBound"] = None
        return lastLeverage


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

class HypothesisVariation:
    def __init__(self, function, **kwargs):
        assert callable(function)
        self.kwargs = kwargs

    def run(self, shortTerm, longTerm, cash, botcoins, customParameters, chartingParameters):
        return bollingerBandsSafe(shortTerm, longTerm, cash, botcoins, customParameters, chartingParameters, **self.kwargs)


# def bollingerBandStdevVariation(givenStdev):
#     assert isinstance(givenStdev, float) or isinstance(givenStdev, int) or isinstance(givenStdev, Decimal)
#     def bollingerBandWrapper(shortTerm, longTerm, cash, botcoins, customParameters, chartingParameters):
#         return bollingerBandsSafe(shortTerm, longTerm, cash, botcoins, customParameters, chartingParameters, bollinger_number_of_stdev=givenStdev)
#     return bollingerBandWrapper
