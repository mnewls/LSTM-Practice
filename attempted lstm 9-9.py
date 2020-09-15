import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import sys

df = pd.read_csv(r'C:\Users\Michael\Desktop\Python\pwrball_rand\pwr_ball - Copy.csv')

trim = df.drop(['prize', 'daysin','daycos','year'], axis=1)

sequence = trim.values.reshape(-1,1).tolist()

ohe = OneHotEncoder().fit(sequence)

encoded_trim = ohe.transform(sequence).toarray()




model = Sequential()
model.add(LSTM(207, input_shape = (6,69), stateful=True, batch_input_shape = (1,6,69)))
#model.add(LSTM(276, input_shape = (6,69), return_sequences = True, stateful = True))
#model.add(LSTM(207, input_shape = (6,69), stateful = True))
#model.add(LSTM(138, input_shape = (6,69), return_sequences = True, stateful = True))
#model.add(LSTM(69, input_shape = (6,69)))
#model.add(tf.keras.layers.Dense(69, activation='relu'))
model.add(tf.keras.layers.Dense(138, activation='relu'))
model.add(tf.keras.layers.Dense(69, activation='softmax'))


model.compile(optimizer=keras.optimizers.Adamax(lr=.01), loss="CategoricalCrossentropy", metrics=['acc'])

#model.build(input_shape=(1,6,69))

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




#17436
def gen_sample(num_start):

    sample_X = encoded_trim[num_start:num_start + 6, :]
    sample_Y = encoded_trim[num_start+6:num_start+7, :]

    sample_X = sample_X.reshape(1,6,69)
    
    return sample_X, sample_Y

accuracy = []
losses = []

for i in range(0,2906, 6):
    X, y = gen_sample(i)

    #batch size is how many "chunks" of (1,6,69) data the model can take per epoch run before it "resets" tensor parameters to learn further
    #epoch is how many runs through the batch dataset that the model gets to learn on, updating the loss each time
    

    history = model.fit(X,y, batch_size=1, shuffle=False ,epochs=100, verbose=2)

    #print(history.history.keys())

    this_acc = history.history['acc']
    this_loss = history.history['loss']

    accuracy += this_acc
    losses += this_loss


plt.subplot(211)
plt.plot(accuracy)
plt.subplot(212)
plt.plot(losses)
plt.show()

pred = []
truth = []

def gen_test_sample(num_start_test):
    test_X = encoded_tests[num_start_test:num_start_test + 6, :]
    test_Y = encoded_tests[num_start_test+6:num_start_test+7, :]

    test_X = test_X.reshape(1,6,69)
    
    return test_X, test_Y

test_acc = []
test_loss = []

for i in range(0,120,6):
    test_x, test_y = gen_test_sample(i)
    this_test_out = model.predict(test_x)

    sample_x = test_x.reshape(6,69)
    sample_x = ohe.inverse_transform(sample_x)


    y_hat = ohe.inverse_transform(this_test_out)


    y_true = ohe.inverse_transform(test_y)


    pred.append(np.int(y_hat))
    truth.append(np.int(y_true))


print("pred: ")
print(pred)
print("true: ")
print(truth)

matches = (np.array(pred) == np.array(truth))

y_match = np.array(truth)[matches]
print(y_match)

num_true = np.count_nonzero(matches)
print(num_true)
