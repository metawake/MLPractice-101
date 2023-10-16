import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.preprocessing import scale
from pandas import pandas

def read_data():
    data = pandas.read_csv('wine.data')
    return data

initial_data =  read_data()
total = len(initial_data)

classes = initial_data[initial_data.columns[0]]
data = initial_data[initial_data.columns[1:]]
data = scale(data)

kf = KFold(total, n_folds=5, random_state=42, shuffle=True)
all_scores = {}
max_values = (0, 0)

for k in range(1, 51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(data, classes)
    print("score", neigh.score(data, classes))
    neigh = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(neigh, data, classes, cv=kf)
    all_scores[k] = sum(scores)/len(scores)
    print(k, sum(scores)/len(scores))
    if all_scores[k] > max_values[1]:
        max_values = k, all_scores[k]

print(max_values)

