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
import sherpa

def gen_sample(num_start_sample, sample_len, train_samples):
    end_of_sample = sample_len + 1

    #print(num_start_sample)
    #print(sample_len)

    train_x = train_samples[num_start_sample:num_start_sample + sample_len, :]
    train_y = train_samples[num_start_sample+sample_len:num_start_sample+end_of_sample, :]

    train_x = train_x.reshape(1,sample_len,69)
    
    return train_x, train_y

def gen_test(num_start_sample, sample_len, test_samples):
    end_of_sample = sample_len + 1

    test_x = test_samples[num_start_sample:num_start_sample + sample_len, :]
    test_y = test_samples[num_start_sample+sample_len:num_start_sample+end_of_sample, :]

    test_x = test_x.reshape(1,sample_len,69)

    return test_x, test_y

#samples, time steps, features
#sample_lens = list(range(6,510,6))

parameters = [sherpa.Discrete('layer_one_dep', [69, 138, 207, 276]),
                sherpa.Ordinal('sample_len', range=[6,12,18,24,30,36,42,48,54,60,66,72,78,84]),
                sherpa.Continuous('learn_rate', range=[.00001, 0.5], scale = 'log'),
                sherpa.Discrete('epoch_num', range=[1,1000], scale='log'),
                sherpa.Discrete('batch_num', range=[2,4,8,16,32,64,128])]
alg = sherpa.algorithms.RandomSearch(max_num_trials=30)

study = sherpa.Study(parameters=parameters, algorithm=alg, lower_is_better=False, disable_dashboard=True, output_dir=r'C:\Users\Michael\Desktop\Python\pwrball_rand\output')

#layer_one_dep, layer_two_dep, sample_len, learn_rate, epoch_num, batch_num

df = pd.read_csv(r'C:\Users\Michael\Desktop\Python\pwrball_rand\pwrball_9_15.csv')
sequence = df.values.reshape(-1,1).tolist()
ohe = OneHotEncoder().fit(sequence)

encoded_sequence = ohe.transform(sequence).toarray()
train, test = train_test_split(encoded_sequence, test_size = 0.05)
counter = 1

for trial in study:

    for i in range(int(len(train)-(trial.parameters['sample_len']+1))):

        #print(trial.parameters['sample_len'])

        x_train, y_train = gen_sample(i, trial.parameters['sample_len'], train)

        model = Sequential()
        model.add(LSTM(trial.parameters['layer_one_dep'], batch_input_shape=(1, trial.parameters['sample_len'], 69), stateful=True))
        #model.add(LSTM(layer_two_dep, batch_input_shape=(1, sample_len, 69), stateful=True))
        model.add(Dense(69, activation='softmax'))

        opt = keras.optimizers.Adam(learning_rate = trial.parameters['learn_rate'])
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

        model.fit(x_train, y_train, epochs=trial.parameters['epoch_num'], batch_size=trial.parameters['batch_num'], verbose=1)

    pred = []
    truth = []
        
    for i in range(int(len(test)-(trial.parameters['sample_len']+1))):

        test_x, test_y = gen_test(i, trial.parameters['sample_len'], test)

        yhat = model.predict(test_x, batch_size=trial.parameters['batch_num'])
        
        y_pred = ohe.inverse_transform(yhat)
        y_true = ohe.inverse_transform(test_y)
    
        pred.append(np.int(y_pred))
        truth.append(np.int(y_true))
    
    
    matches = (np.array(pred) == np.array(truth))

    per_true = (np.count_nonzero(matches) / len(truth)) * 100

    print('trial complete @ ' + counter)
    print(per_true)

    counter+=1

    study.finalize(trial)

    

#print(check_test)



    
