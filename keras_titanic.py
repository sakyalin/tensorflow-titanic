import numpy as np
import pandas as pd

# load data
train_data = pd.read_csv(r"./train.csv")
test_data = pd.read_csv(r"./test.csv")


def drop_not_concerned_columns(data, columns):
    return data.drop(columns, axis=1)


not_concerned_columns = ["PassengerId", "Name", "Ticket", "Fare", "Cabin", "Embarked"]
train_data = drop_not_concerned_columns(train_data, not_concerned_columns)
test_data = drop_not_concerned_columns(test_data, not_concerned_columns)


def clean_nan_columns(data, columns):
    data.dropna()
    return data


nan_columns = ["Age", "SibSp", "Parch"]

train_data = train_data.dropna()
print(len(train_data))
test_data = test_data.dropna()
print(len(test_data))


# normalize
def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


dummy_columns = ["Pclass"]
train_data = dummy_data(train_data, dummy_columns)
test_data = dummy_data(test_data, dummy_columns)

from sklearn.preprocessing import LabelEncoder


def sex_to_int(data):
    le = LabelEncoder()
    le.fit(["male", "female"])
    data["Sex"] = le.transform(data["Sex"])
    return data


train_data = sex_to_int(train_data)
test_data = sex_to_int(test_data)
train_data.head()

from sklearn.preprocessing import StandardScaler


def normalize_age(data):
    ss = StandardScaler()
    data["Age"] = ss.fit_transform(data["Age"].values.reshape(-1, 1))
    return data


train_data = normalize_age(train_data)
test_data = normalize_age(test_data)
train_data.head()


def split_valid_test_data(data, fraction=0.8):
    data_y = data["Survived"]
    data_x = data.drop(["Survived"], axis=1)

    train_valid_split_idx = int(len(data_x) * fraction)
    train_x = data_x[:train_valid_split_idx]
    train_y = data_y[:train_valid_split_idx]

    valid_test_split_idx = (len(data_x) - train_valid_split_idx) // 2
    test_x = data_x[train_valid_split_idx + valid_test_split_idx:]
    test_y = data_y[train_valid_split_idx + valid_test_split_idx:]

    return train_x.values, train_y.values.reshape(-1, 1), test_x.values, test_y.values.reshape(-1, 1)


train_x, train_y, test_x, test_y = split_valid_test_data(train_data)
print("train_x:{}".format(train_x.shape))
print("train_y:{}".format(train_y.shape))

print("test_x:{}".format(test_x.shape))
print("test_y:{}".format(test_y.shape))

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam

model = Sequential()
model.add(Dense(20, input_dim=train_x.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, input_dim=20))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.0001)
model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(train_x, train_y, nb_epoch=12000, batch_size=32)
score = model.evaluate(test_x, test_y)
print("")
print("Test loss:{0}".format(score[0]))
print("Test accuracy:{0}".format(score[1]))
