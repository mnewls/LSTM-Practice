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


df = pd.read_csv(r'C:\Users\Michael\Desktop\Python\pwrball_rand\pwrball_9_15.csv')

#print(df.head)

sequence = df.values.reshape(-1,1).tolist()
#print(sequence)

ohe = OneHotEncoder().fit(sequence)

encoded_sequence = ohe.transform(sequence).toarray()

#print(encoded_sequence)

#samples, time steps, features

train, test = train_test_split(encoded_sequence, test_size = 0.05)

#print(train.shape)

def gen_sample(num_start_sample):
    train_x = train[num_start_sample:num_start_sample + 12, :]
    train_y = train[num_start_sample+12:num_start_sample+13, :]

    train_x = train_x.reshape(1,12,69)
    
    return train_x, train_y

def gen_test(num_start_sample):
    test_x = test[num_start_sample:num_start_sample + 12, :]
    test_y = test[num_start_sample+12:num_start_sample+13, :]

    test_x = test_x.reshape(1,12,69)

    return test_x, test_y


#samples, time steps, features

model = Sequential()
model.add(LSTM(69, batch_input_shape=(1, 12, 69), return_sequences=True, stateful=True))
model.add(LSTM(69, batch_input_shape=(1, 12, 69), stateful=True))
model.add(Dense(69, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate = 0.00001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

model.summary()
# fit model

accuracy = []
losses = []

for i in range(len(train)-13):

    this_x, this_y = gen_sample(i)

    check_test = ohe.inverse_transform(this_y)

    #print(check_test)

    history = model.fit(this_x, this_y, epochs=100, batch_size=256, verbose=2, shuffle=False)

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

# evaluate model on new data

for i in range(len(test)-13):

    this_x, this_y = gen_test(i)

    yhat = model.predict(this_x, batch_size=5)

    y_pred = ohe.inverse_transform(yhat)
    y_true = ohe.inverse_transform(this_y)
   
    pred.append(np.int(y_pred))
    truth.append(np.int(y_true))


matches = (np.array(pred) == np.array(truth))

print(len(truth))

print(pred)

y_match = np.array(truth)[matches]
print(y_match)

per_true = (np.count_nonzero(matches) / len(truth)) * 100
print(per_true)