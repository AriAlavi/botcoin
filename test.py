from main import convertData, BotCoin
from datetime import datetime, timedelta
STUPID = "stupid.csv" 

def test_FetchData():
    bot = BotCoin(STUPID)
    start = datetime(year=2010, month=4, day=20, hour=1)
    gotData = bot.fetchData(start, timedelta(minutes=1))
    print(len(gotData))
    assert(len(gotData), 1000)
    assert(len(bot.fetchData(start, timedelta(days=365))), 1)

