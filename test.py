from main import *
from dataTypes import *

from datetime import datetime, timedelta
from decimal import Decimal
from main import convertData, BotCoin

STUPID = "stupid.csv" 

def test_FetchData1():
    bot = BotCoin(STUPID)
    start = datetime(year=2010, month=4, day=19, hour=1)
    gotData = bot.fetchData(start, timedelta(minutes=1))

    assert len(gotData) == 0
    assert len(bot.fetchData(start, timedelta(days=999))) == 1000
    assert len(bot.fetchData(start, timedelta(days=6))) == 458
    assert len(bot.fetchData(start, timedelta(hours=22))) == 19

def test_FetchData2():
    bot = BotCoin(STUPID)
    start = datetime(year=2010, month=5, day=2, hour=1) 
    
    assert len(bot.fetchData(start, timedelta(days=999))) == 0
    assert len(bot.fetchData(start, timedelta(days=5))) == 0
    assert len(bot.fetchData(start, timedelta(hours=6))) == 0

def test_FetchData3():
    bot = BotCoin(STUPID)
    start = datetime(year=2010, month=4, day=26, hour=1) 
    
    assert len(bot.fetchData(start, timedelta(days=4))) ==346
    assert len(bot.fetchData(start, timedelta(days=6))) == 456
    assert len(bot.fetchData(start, timedelta(hours=6))) == 22

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