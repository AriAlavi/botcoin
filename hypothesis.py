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