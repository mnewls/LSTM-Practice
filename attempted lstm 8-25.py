import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from pandas.plotting import autocorrelation_plot
from keras import Sequential
from tensorflow.python.keras.layers.recurrent import LSTM



df = pd.read_csv(r'C:\Users\Michael\Desktop\pwrball_rand\pwr_ball - Copy.csv')

#print(df.head(5))

#df.hist(bins = 1)

#autocorrelation_plot(df)

'''def show_heatmap(data):
    plt.matshow(df.corr())
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()'''


#show_heatmap(df)

split_fraction = 0.715
train_split = int(split_fraction * int(df.shape[0]))
step = 1

future = 1
learning_rate = 0.001
batch_size = 10
epochs = 10


def normalize(data, train_split):
    data_mean = data[:train_split].mean(axis=0)
    data_std = data[:train_split].std(axis=0)
    return (data - data_mean) / data_std

#print(data_in.head())

features = normalize(df, train_split)
features = pd.DataFrame(features)
'''print(features.head)
print(features.shape)'''

train_data = features.loc[0 : train_split - 1]
val_data = features.loc[train_split:]

#print(val_data.shape)

'''print("")
print(train_data.shape)
print(train_data.head)
print("")'''

train_x = train_data.iloc[:, 1:]

train_y = train_data.iloc[:, 0]



train_y = pd.DataFrame(train_y)
#print(train_y)

train_y.wb1 = train_y.wb1.shift(1)
#print(train_y)
val_x = val_data.iloc[:,1:]
val_y = val_data.iloc[:,0]

'''print("")
print(train_y.shape)
print(train_y.head)
print("")'''

model = Sequential()
model.add(LSTM(2077, input_shape = (2077,9), return_sequences = True))
model.add(LSTM(2077, input_shape = (2077,9), return_sequences = True))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='relu'))


model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
model.summary()

path_checkpoint = "model_checkpoint.h5"
es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)


#print(train_x.shape)

modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_loss",
    filepath=path_checkpoint,
    verbose=1,
    save_weights_only=True,
    save_best_only=True,
)

train_x = train_x.values.reshape(-1,2077,9)

train_y = train_y.values.reshape(-1,2077,1)

val_x = val_x.values.reshape(-1,829,9)
val_y = val_y.values.reshape(-1,829,1)

history = model.fit(
    train_x,
    train_y,
    epochs=epochs,
    validation_data=(val_x, val_y)
)

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


visualize_loss(history, "Training and Validation Loss")


print(model.predict(train_x))

def de_normalize(data):
    data_mean = df.mean(axis=0)[0]
    data_std = df.std(axis=0)[0]
    return ((data + data_mean) * data_std)

print(de_normalize(model.predict(train_x)))