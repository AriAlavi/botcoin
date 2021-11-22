from dataTypes import *
from decimal import Decimal

def randomChoice(shortTerm, longTerm, cash, botcoins, customParameters):
    assert isinstance(shortTerm, DiscreteData)
    assert isinstance(longTerm, DiscreteData)
    assert isinstance(cash, Decimal)
    assert isinstance(botcoins, Decimal)
    assert isinstance(customParameters, dict)
    import random
    return Decimal(random.randint(0, 100))/100

def bounce(shortTerm, longTerm, cash, botcoins, customParameters):
    assert isinstance(shortTerm, DiscreteData)
    assert isinstance(longTerm, DiscreteData)
    assert isinstance(cash, Decimal)
    assert isinstance(botcoins, Decimal), "{} instead".format(type(botcoins))
    assert isinstance(customParameters, dict)

    sellAll = customParameters.get("sell", False)
    if sellAll:
        customParameters["sell"] = False
        return Decimal(0)
    else:
        customParameters["sell"] = True
        return Decimal(1)

def hold(*args):
    return Decimal(1)

def bollingerBands(shortTerm, longTerm, cash, botcoins, customParameters):
    assert isinstance(shortTerm, DiscreteData)
    assert isinstance(longTerm, DiscreteData)
    assert isinstance(cash, Decimal)
    assert isinstance(botcoins, Decimal), "{} instead".format(type(botcoins))
    assert isinstance(customParameters, dict)

    BOLLINGER_BAND_TIME_PERIOD = 20

    if not isinstance(customParameters.get("history", None), list):
        customParameters["history"] = []
    
    if len(customParameters["history"]) == 0:
        customParameters["history"].append(longTerm)
    else:
        if longTerm.date != customParameters["history"][-1].date:
            customParameters["history"].append(longTerm)
        if len(customParameters["history"]) > BOLLINGER_BAND_TIME_PERIOD:
            customParameters["history"].pop(0)

    # print(len(customParameters["history"]))

    return Decimal(1)