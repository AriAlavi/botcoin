from pandas import pd
dataframe = pd.read_excel("ETHUSD.csv")
pd.DataFrame(wholedf[3000000:]).to_excel("splitethusd.xlsx")  