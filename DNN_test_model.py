import tensorflow as tf
from keras import Sequential, layers
from keras.utils import np_utils
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv('./csvfile/featrue.csv')
# df1 = pd.DataFrame(df)
# print(df1)
dataset = df.values
# print(dataset)
dataset = np.array(dataset)
# print(dataset)
# print(dataset.ndim)
x = dataset[:, 0:8]        # 資料
y = dataset[:, 8]          # 標籤
# print(x)
# print(y)
mm_scale = preprocessing.MinMaxScaler()
x_scale = mm_scale.fit_transform(x)
# print(x_training)
lb = LabelEncoder()
y = lb.fit_transform(y)               # 將label類別轉換為數字型態(如:0、1、2.....)
y = np_utils.to_categorical(y)
# print(y.shape)
x_train, x_val_and_test, y_train, y_val_and_test = train_test_split(x_scale, y, test_size=0.3)
# print(x_train.shape[1])
# train_test_split()功能只能將資料拆分成兩分，test_size->第二份資料拆成30%
# x_train:訓練集；x_val_and_test:驗證與測試集
x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=0.5)
# 再使用train_test_split()將 x_val_and_test 拆分成> x_val:驗證集；x_test:測試集，將兩份資料拆成對半

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
# x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)

model = Sequential([
    layers.Dense(256, activation="relu", input_dim=8),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax")
])

# model.summary()

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=500, epochs=50, shuffle=100, validation_data=(x_val, y_val), verbose=2)

model.evaluate(x_test, y_test)

model.save("my_model.h5")
