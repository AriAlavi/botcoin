from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from tracemalloc import start
import urllib.request

from numpy import isin

def generateLinks(startDate, endDate):
    assert isinstance(startDate, datetime)
    assert isinstance(endDate, datetime)
    # https://files.pushshift.io/reddit/submissions/RS_2016-02.zst
    links = []
    while startDate <= endDate:
        strDate = startDate.strftime("%Y-%m")
        links.append(f"https://files.pushshift.io/reddit/submissions/RS_{strDate}.zst")
        startDate += relativedelta(months=1)
    return links

def downloadFromLink(givenLink, filename):
    assert isinstance(givenLink, str)
    assert isinstance(filename, str)
    assert "http" in givenLink

    print(f"Downloading from '{givenLink}'...")
    urllib.request.urlretrieve(givenLink, filename)
    print(f"Download from '{givenLink}' complete.")
    # with urllib.request.urlopen(givenLink) as f:
    #     html = f.read().decode('utf-8')
    #     file = open(filename, "wb")
    #     file.writelines(html)
    #     file.close()


def downloadPushshift(givenLink):
    assert isinstance(givenLink, str)
    filename = givenLink.split("/")[-1]
    downloadFromLink(givenLink, filename)

for link in generateLinks(datetime(year=2015, month=1, day=1), datetime(year=2021, month=6, day=1)):
    downloadPushshift(link)