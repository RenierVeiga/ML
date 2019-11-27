import quandl
import pandas as pd
import numpy as np
from os import path
from sklearn.preprocessing import MinMaxScaler


symbol = "EOD/HD"
authtoken = "dqXw1Ydw-nzAN2nSTMaP"


def load_data(start_date=""):
    file = symbol + ".csv"
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


df, last_date = load_data()


def do_delta(data):
    diff = []
    end = len(data)
    delta = data[0]
    for i in range(1, end):
        diff.append(data[i] - data[i - 1])
    return diff, delta


def undo_delta(data, delta):
    diff = []
    end = len(data)
    diff.append(delta)
    for i in range(1, end):
        diff.append(data[i] + diff[i - 1])
    return diff


print(df.head())
features_considered = ['Adj_Close', 'Adj_Volume']
features = df[features_considered]
dataset = features.values
diff, delta = do_delta(dataset)

print("Diff series: ")
print(diff[0])

scaler = MinMaxScaler()
normalized = scaler.fit_transform(diff)

print("Normalized series: ")
print(normalized[0])

unnorm = scaler.inverse_transform(normalized)
print("UnNormalized series: ")
print(unnorm[0])

undif = undo_delta(unnorm, delta)

print("Verify Undo Delta:")
print("Is Delta " + delta[0].__str__() + " == UnDiff " + undif[0][0].__str__())


# Prepare data for model
def training_val_data(dataset, target, offset=0):
    data = []
    labels = []

    start_index = offset
    end_index = len(dataset) - (window_size + future_target)

    for i in range(start_index, end_index, 1):
        j = i + window_size
        temp_data = dataset[i:j]
        if len(temp_data) == window_size:
            temp_label = target[j:j + future_target]
            if len(temp_label) == future_target:
                data.append(temp_data)
                labels.append(temp_label)

    return np.array(data), np.array(labels)