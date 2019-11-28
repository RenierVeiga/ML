from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from os import path


def load_model(window_size, future_target, train_dataset, val_dataset, fileName):
    file = fileName + window_size.__str__() + ".h5"

    if path.exists(file):
        model = tf.keras.models.load_model(file)
        print("Model exists")
    else:
        print("Model does not exist")
        # Nh=Ns/(α∗(Ni+No)) Ni = number of input neurons. No = number of output neurons. Ns = number of samples in training data set. α = an arbitrary scaling factor usually 2-10.
        alpha = 2
        ni = train_dataset[0].shape[1]
        ns = train_dataset[0].shape[0]
        no = future_target
        # nh = int(ns / (alpha * (ni + no)))
        nh = 128
        print("nh: " + nh.__str__())

        model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(nh, return_sequences=True, input_shape=train_dataset[0].shape[1:]),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.SimpleRNN(nh),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(nh, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(future_target, activation="linear")])
        model.compile(optimizer=tf.optimizers.Adam(lr=0.001, decay=1e-7), loss='mse', metrics=['accuracy'])

    return model_fit(model, fileName, train_dataset, val_dataset)


def model_fit(model, file, train_dataset, val_dataset):
    callbacks = [
        keras.callbacks.EarlyStopping(
            # Stop training when `val_loss` is no longer improving
            monitor='val_loss',
            mode='min',
            # "no longer improving" being defined as "no better than 1e-2 less"
            min_delta=1e-7,
            # "no longer improving" being further defined as "for at least 2 epochs"
            patience=6,
            restore_best_weights=True,
            verbose=1)
    ]

    model.fit(train_dataset, epochs=1000, callbacks=callbacks,
              validation_data=val_dataset, validation_split=0.8, verbose=2)

    model.save(file)
    return model
