import quandl
import pandas as pd
import numpy as np
from os import path
from sklearn.preprocessing import MinMaxScaler


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


def do_delta(data):
    diff = []
    end = len(data)
    delta = data[0]
    for i in range(1, end):
        diff.append(data[i] - data[i - 1])
    print("Diff series: ")
    print(diff[0])
    return diff, delta


def undo_delta(data, delta):
    diff = []
    end = len(data)
    diff.append(delta)
    for i in range(1, end):
        diff.append(data[i] + diff[i - 1])
    print("Undo Diff Series:")
    print("Is Delta " + delta[0].__str__() + " == UnDiff " + diff[0][0].__str__())
    return diff


def norm(data):
    scalar = MinMaxScaler()
    normalized = scalar.fit_transform(data)
    print("Normalized series: ")
    print(normalized[0])
    return scalar, normalized


def unnorm(scalar, normalized):
    unnorm = scalar.inverse_transform(normalized)
    print("UnNormalized series: ")
    print(unnorm[0])
    return unnorm


df, last_date = load_data("FB")
print(df.head())
features_considered = ['Adj_Close', 'Adj_Volume']
features = df[features_considered]
dataset = features.values
diff, delta = do_delta(dataset)
scalar, norm = norm(diff)
unnorm = unnorm(scalar, norm)
undif = undo_delta(unnorm, delta)


# Prepare data for model
def subsample_data(dataset, label_index, window_size, future_target):
    data = []
    labels = []
    target = dataset[:, label_index]
    start_index = window_size
    end_index = len(dataset) - future_target

    for i in range(start_index, end_index):
        data.append(dataset[i - window_size:i])
        labels.append(target[i:i + future_target])

    return np.array(data), np.array(labels)
