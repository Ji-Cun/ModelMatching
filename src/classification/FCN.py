# -*- coding: utf-8 -*-

from tensorflow import keras
import numpy as np
import pandas as pd

class FCN:
    def __init__(self,input_shape,nb_classes):
        self.model=self.build_model(input_shape,nb_classes)
        self.p=0

    def build_model(self,input_shape,nb_classes):

        x = keras.layers.Input(input_shape)
        #    drop_out = Dropout(0.2)(x)
        conv1 = keras.layers.Conv1D(128, 8, 1, padding='same')(x)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)

        #    drop_out = Dropout(0.2)(conv1)
        conv2 = keras.layers.Conv1D(256, 5, 1, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        #    drop_out = Dropout(0.2)(conv2)
        conv3 = keras.layers.Conv1D(128, 3, 1, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        full = keras.layers.GlobalAveragePooling1D()(conv3)
        out = keras.layers.Dense(nb_classes, activation='softmax')(full)


        model = keras.models.Model(inputs=x, outputs=out)

        optimizer = keras.optimizers.Nadam()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def fit(self, x_train, Y_train, nb_epochs=50):
        batch_size = int(min(x_train.shape[0]/10, 16))

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.001)
        hist=self.model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                              verbose=0, callbacks = [reduce_lr])
        log = pd.DataFrame(hist.history)
        acc=log.iloc[-1]['accuracy']
        self.p=acc

    def predict(self,x_test):
        y_pred=self.model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred



