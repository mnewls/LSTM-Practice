import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras
from pandas.plotting import autocorrelation_plot
from keras import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from random import randint
import sys

df = pd.read_csv(r'C:\Users\Michael\Desktop\pwrball_rand\pwr_ball - Copy.csv')

trim = df.drop(['prize', 'daysin','daycos','year'], axis=1)

#print(trim)

sequence = trim.values.reshape(-1,1).tolist()
#print(sequence)

ohe = OneHotEncoder().fit(sequence)

encoded_trim = ohe.transform(sequence).toarray()

#np.set_printoptions(threshold=sys.maxsize)

row, col = encoded_trim.shape

def gen_sample(num_start):
    #start_of_sample = randint(0,17436)
    #print(start_of_sample)
    sample_X = encoded_trim[num_start:num_start + 6, :]
    sample_Y = encoded_trim[num_start+6:num_start+7, :]

    sample_X = sample_X.reshape(1,6,69)
    #sample_Y = sample_Y.reshape(1,1,69)

    #print(sample_X.shape)
    #print(sample_Y.shape)
    
    return sample_X, sample_Y

#this_x, this_y = gen_sample(0)

model = Sequential()
model.add(LSTM(138, input_shape = (6,69), return_sequences = True))
model.add(LSTM(69, input_shape = (6,69)))
model.add(tf.keras.layers.Dense(69, activation='softmax'))


model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05), loss="categorical_crossentropy")
model.summary()

test_num = [9,36,49,56,62,9]
#june 27

test_num = np.asarray(test_num)
test_num = test_num.reshape(-1,1)

test_num_encode = ohe.transform(test_num).toarray()
#print(test_num_encode)
test_sample = test_num_encode.reshape(1,6,69)


for i in range(17429):
    X, y = gen_sample(i)
    model.fit(X,y,epochs=1, verbose=2)
    #model.reset_states()

test_out = model.predict(test_sample)

test_out = ohe.inverse_transform(test_out)

print(test_out)
#expect 15

#test nums:
#9,36,49,56,62,8

#lstm_input_dataframe = pd.DataFrame(np.concatenate(lstm_input_unsplit))

#decoded_trim = ohe.inverse_transform(encoded_trim)

#print(type(decoded_trim))