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
#print(row)
#print(col)

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
#model.add(LSTM(276, input_shape = (6,69), return_sequences = True))
#model.add(LSTM(276, input_shape = (6,69), return_sequences = True))
model.add(LSTM(276, input_shape = (6,69), return_sequences = True))
#model.add(LSTM(138, input_shape = (6,69), return_sequences = True))
model.add(LSTM(207, input_shape = (6,69)))
model.add(tf.keras.layers.Dense(138, activation='relu'))
#model.add(tf.keras.layers.Dense(69, activation='swish'))
model.add(tf.keras.layers.Dense(69, activation='softmax'))


model.compile(optimizer=keras.optimizers.Adagrad(lr = .0001), loss="categorical_crossentropy")
model.summary()

test_1 = [9,36,49,56,62,9]
#june 27
test_2 = [15,28,52,53,63,18]
#july 1
test_3 = [16,21,27,60,61,6]
#july4
test_4 = [3,10,34,36,62,5]
#july 8
test_5 = [14,19,61,62,64,4]
#july11
test_6 = [27,47,61,62,69,4]
#july15
test_7 = [13,16,32,58,59,9]
#july18
test_8 = [16,25,36,44,55,14]
#july22
test_9 = [5,21,36,61,62,18]
#july 25
test_10 = [7,29,35,40,45,26]
#july29
test_11 = [6,25,36,43,48,24]
#aug 1
test_12 = [7,14,17,57,65,24]
#aug 5
test_13 = [2,3,14,40,51,24]
#aug8
test_14 = [2,6,18,36,37,21]
#aug12
test_15 = [5,12,34,45,56,3]
#aug15
test_16 = [13,23,47,55,58,23]
#aug19
test_17 = [19,30,36,42,66,14]
#aug22
test_18 = [8,12,19,47,58,2]
#aug 26
test_19 = [5,21,22,29,43,10]
#aug29
test_20 = [1,4,11,20,69,18]
#sept 2
test_21 = [15,21,22,27,47,7]
#sept 5

tests = test_1 + test_2+ test_3+ test_4+ test_5+ test_6+ test_7+ test_8+ test_9+ test_10+ test_11+ test_12+ test_13+ test_14+ test_15+ test_16+ test_17+ test_18+ test_19+ test_20+ test_21

tests = pd.DataFrame(tests)

test_sequence = tests.values.reshape(-1,1).tolist()

encoded_tests = ohe.transform(test_sequence).toarray()

#print(encoded_tests.shape)

#print(tests)

def gen_test_sample(num_start_test):
    test_X = encoded_tests[num_start_test:num_start_test + 6, :]
    test_Y = encoded_tests[num_start_test+6:num_start_test+7, :]

    test_X = test_X.reshape(1,6,69)
    
    return test_X, test_Y

#17436

for i in range(0,2906, 6):
    X, y = gen_sample(i)
    #y_sample = ohe.inverse_transform(y)

    #sample_x = X.reshape(6,69)

    #sample_x = ohe.inverse_transform(sample_x)
    #print(sample_x)

    #print(X)
    #print(y_sample)
    model.fit(X,y, batch_size=2, shuffle=False ,epochs=50, verbose=2)
    #model.reset_states()


pred = []
truth = []
#print(type(pred))

for i in range(0,120,6):
    test_x, test_y = gen_test_sample(i)
    this_test_out = model.predict(test_x)

    sample_x = test_x.reshape(6,69)
    sample_x = ohe.inverse_transform(sample_x)

    #print(sample_x)

    y_hat = ohe.inverse_transform(this_test_out)

    #print(type(y_hat))

    y_true = ohe.inverse_transform(test_y)

    #print(type(y_true))

    #print(type(temp))
    pred.append(np.int(y_hat))
    truth.append(np.int(y_true))


print("pred: ")
print(pred)
print("true: ")
print(truth)

matches = (np.array(pred) == np.array(truth))

#print(matches)
#print(type(matches))
#print(type(truth))

y_match = np.array(truth)[matches]
print(y_match)

num_true = np.count_nonzero(matches)
print(num_true)


#test_out = model.predict(test_sample)

#test_out = ohe.inverse_transform(test_out)

#print(test_out)
#expect 15

#test nums:
#9,36,49,56,62,8

#lstm_input_dataframe = pd.DataFrame(np.concatenate(lstm_input_unsplit))

#decoded_trim = ohe.inverse_transform(encoded_trim)

#print(type(decoded_trim))