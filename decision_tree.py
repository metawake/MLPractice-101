from sklearn.tree import DecisionTreeClassifier
from pandas import pandas

def read_data():
    COLUMNS = ['Pclass','Fare','Age','Sex', 'Survived']
    data = pandas.read_csv('titanic.csv', usecols=COLUMNS)
    return data


def prepare_data(data):
    data_cleaned = data[pandas.notnull(data['Age'])]
    data_mf = data_cleaned.replace('female', False).replace('male', True)
    return data_mf


def get_features(data):
    FIELDS = ['Pclass','Sex','Fare','Age']
    TARGET_FIELD = ['Survived']
    clf = DecisionTreeClassifier()
    clf.fit(data[FIELDS], data[TARGET_FIELD])
    importances = clf.feature_importances_
    return list(zip(['Pclass','Sex','Fare','Age'], importances))


base_data = read_data()
prepared_data = prepare_data(base_data)

print(get_features(prepared_data))