import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


# features_considered = ['Adj_Close', 'Adj_Volume']
# features = df[features_considered]
# dataset = features.values
# diff, delta = do_delta(dataset)
# scalar, norm = norm(diff)
# unnorm = unnorm(scalar, norm)
# undif = undo_delta(unnorm, delta)