from keras import models, layers, optimizers
from keras.regularizers import L1L2
from keras.callbacks import EarlyStopping


def initialize_model(X_train, y_train):

    reg = regularizers.l1_l2(l1=0.005)

    # Architecture
    model = models.Sequential()
    model.add(layers.LSTM(8, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=reg))
    model.add(layers.Dropout(rate=0.1))
    output_len = y_train.shape[1]
    model.add(layers.Dense(output_len, activation='linear'))

    # Compile
    adam = optimizers.Adam(learning_rate=0.05)

    model.compile(loss='mse',
                 optimizer=adam,
                 metrics=['mae'])

    return model

def fit_model(model):

    es = EarlyStopping(patience=10, restore_best_weights=True, monitor='val_mae', mode='min')

    history = model.fit(X_train, y_train,
                        validation_split=0.3,
                        shuffle = False,
                        batch_size=16,
                        epochs=50,
                        verbose=1,
                        callbacks=[es]
                       )

    return (model, history)
