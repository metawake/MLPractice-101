from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from numpy import linspace

def load_data():
    return load_boston()


def main():

    data = load_data()
    data.data = scale(data.data)

    intervals = linspace(1,10, num=200)

    all_scores = {}
    max_values = 0, 0

    for p in intervals:

        neigh = KNeighborsRegressor(n_neighbors=5, weights='distance',p=p)
        neigh.fit(data.data, data.target)

        kf = KFold(len(data.data), n_folds=5, random_state=42, shuffle=True)

        scores = cross_val_score(neigh, data.data, data.target, cv=kf)

        all_scores[p] = sum(scores)/len(scores)
        print(p, sum(scores)/len(scores))
        if all_scores[p] > max_values[1]:
            max_values = p, all_scores[p]

    print(max_values)

main()
