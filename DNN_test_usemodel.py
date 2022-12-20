import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.models import load_model

train = np.array([[170, 90, 176, 165, 175, 156, 177, 180]])

model = load_model("my_model.h5")
# model.summary()
pre = model.predict(train)
# if pre == np.array([[0, 1, 0]]):
#     print("true")
print(pre)
