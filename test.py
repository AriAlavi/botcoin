from main import *
from dataTypes import *

from datetime import datetime, timedelta
from decimal import Decimal

STUPID = "stupid.csv" 

def test_FetchData():
    bot = BotCoin(STUPID)
    start = datetime(year=2010, month=4, day=20, hour=1)
    gotData = bot.fetchData(start, timedelta(minutes=1))

    assert len(gotData) == 1000
    assert len(bot.fetchData(start, timedelta(days=365))) == 1

def test_SetLeverage():
    def leverageTester(cash, coins, PRICE, soughtLeverage, goingDown):
        assert isinstance(soughtLeverage, Decimal)
        if goingDown and soughtLeverage < 0:
            return {"cash" : cash, "coins" : coins}
        
        print("SOUGHT", float(soughtLeverage))
        result = setLeverage(cash, coins, PRICE, soughtLeverage)
        if goingDown:
            newLeverage = soughtLeverage - Decimal(".1")
        else:
            if soughtLeverage == Decimal(1):
                return leverageTester(result["cash"], result["coins"], PRICE, Decimal(".9"), True)
            else:
                newLeverage = soughtLeverage + Decimal(".1")
        return leverageTester(result["cash"], result["coins"], PRICE, newLeverage, goingDown)

    result = leverageTester(Decimal("1000"), Decimal("0"), Decimal(1), Decimal(0), False)
    assert result["cash"] == Decimal("1000")

    result = leverageTester(Decimal("0"), Decimal("1000"), Decimal(1), Decimal(0), False)
    assert result["cash"] == Decimal("1000")

    result = leverageTester(Decimal("10000"), Decimal("0"), Decimal(10), Decimal(0), False)
    assert result["cash"] == Decimal("10000")

    result = leverageTester(Decimal("0"), Decimal("1000"), Decimal(10), Decimal(0), False)
    assert result["cash"] == Decimal("10000")