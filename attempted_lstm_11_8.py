import pandas as pd
from pandas import DataFrame
import keras
from pandas import concat
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import math

df = pd.read_csv(r'C:\Users\Michael\Desktop\Python\pwrball_rand\cleaned_11_8.csv')
df = df.dropna()

y = df.iloc[:, -1]


#sequence = y.values.reshape(-1,1).tolist()
#ohe = OneHotEncoder().fit(sequence)
#encoded_y = DataFrame(ohe.transform(sequence).toarray())

#cat_data = pd.concat([X, encoded_y], axis=1)

'''cat_data.rename(columns={'Day':'Day', 'Month':'Month', 'Year':'Year', 'Normalized Prize':'Normalized Prize', 'Ball Place Raw':'Ball Place', 'Ball Place Rot','1','2','3','4','5','6', '7','8','9','10',
                            '11', '12', '13', '14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33',
                            '34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57',
                            '58','59','60','61','62','63','64','65','66','67','68','69'}, inplace=True)'''

#print(cat_data.head(5))

train_data_len = math.ceil(len(y) * .85)

#train_set = cat_data.iloc[0:int(train_data_len), :]
#test_set = cat_data.iloc[train_data_len:, :]

#print(type(test_set))
train_set = df.iloc[0:train_data_len, :]
test_set = df.iloc[train_data_len:, :]
 

#print(train_set.head(5))
train_x = train_set.iloc[:, 0:6]
train_y = train_set.iloc[:, 6:]

#print(type(train_y))

#train_y.reshape(len(train_y), )


#print(train_y.head(5))
test_x = test_set.iloc[:, 0:6]
test_y = test_set.iloc[:, 6:]


#test_y.reshape(len(test_y), )

#print(train_x.head(5))
#print(train_y.head(5))

from sklearn.tree import ExtraTreeClassifier
classifier = ExtraTreeClassifier(random_state=0, criterion="entropy", splitter="best")

classifier.fit(train_x, train_y.values.ravel())

info = classifier.score(test_x,test_y.values.ravel())


print(info)
#model = Sequential()
