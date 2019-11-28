import quandl
import pandas as pd
from os import path

dir = "EOD/"
authtoken = "dqXw1Ydw-nzAN2nSTMaP"


def load_data(symbol, start_date=""):
    file = dir + symbol + ".csv"
    last_date = start_date
    if path.exists(file):
        df = pd.read_csv(file, sep="\t")
        last_date = pd.Categorical(df["Date"]).get_values()[-1:][0]

    if start_date != "":
        df.append(quandl.get(symbol, authtoken=authtoken, start_date=start_date), sort=False)
        df.to_csv(file, sep="\t")
    else:
        df = quandl.get(symbol, authtoken=authtoken)
        df.to_csv(file, sep="\t")

    print(last_date)
    return df, last_date