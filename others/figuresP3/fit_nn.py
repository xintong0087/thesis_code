from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, InputLayer, Flatten
from tensorflow.keras.layers import SimpleRNN, LSTM
import tensorflow as tf

import time
import os


def build_model(X,
                architecture="RNN", recurrent_layer_size=(32, 4), dense_layer_size=32,
                activation_func="ReLU",
                lr=0.001, dropout=0.1, decay_rate=0.9):

    if activation_func == "ReLU":
        activation = tf.keras.activations.relu
    else:
        activation = tf.keras.activations.tanh

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr,
                                                                 decay_steps=1000,
                                                                 decay_rate=decay_rate)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)

    model = Sequential()

    if architecture == "RNN":
        model.add(InputLayer(shape=(X.shape[1], 1)))
        model.add(SimpleRNN(units=recurrent_layer_size[0], activation=activation, return_sequences=True))
        model.add(SimpleRNN(units=recurrent_layer_size[1], activation=activation, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(dense_layer_size, activation=tf.keras.activations.relu))

    elif architecture == "LSTM":   
        model.add(InputLayer(shape=(X.shape[1], 1)))
        model.add(LSTM(units=recurrent_layer_size[0], activation=activation, return_sequences=True))
        model.add(LSTM(units=recurrent_layer_size[1], activation=activation, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(dense_layer_size, activation=tf.keras.activations.relu))

    elif architecture == "FFN":
        model.add(InputLayer(shape=(X.shape[1],)))
        model.add(Dense(units=recurrent_layer_size[0], activation=activation))
        model.add(Dropout(dropout))
        model.add(Dense(units=recurrent_layer_size[1],  activation=activation))

    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",
                  optimizer=optimizer, metrics=['mean_squared_error'])

    print(model.summary())

    return model


def train_model(X, y, model, n_epochs, batch_size, patience, save_path):

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    filename = save_path + "training_history.csv"
    logger = tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

    start = time.time()

    history = model.fit(X, y, epochs=n_epochs,
                        batch_size=batch_size,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[callback, logger])

    try:
    
        # Save history plot
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')

        plt.savefig(save_path + "training_history.png")

    except:
        print("Could not save training history plot")

    execution_time = time.time() - start

    print(f"Training time: {execution_time}")

    return model, execution_time
