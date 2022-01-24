import random
import numpy as np
from keras.layers import Input, Dense, LSTM, Lambda
from keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K  

def onehotencoder(lst, n_features):
    arr = []
    vec = []
    for i in range(0, n_features):
        arr.append(0)
    dp = arr.copy()
    for x in lst:
        dp[x] = 1
        vec.append(dp)
        dp = arr.copy()
    vec = np.array(vec)
    return vec

n_units = 16
n_features = 10

#data
X_train = []
y_train = []
X_test = []
y_test = []
for i in range(0, 2000):
    lst = [random.randrange(0,9,1) for i in range(4)]
    rev = list(reversed(lst))
    lst = onehotencoder(np.array(lst), n_features=n_features)
    rev = onehotencoder(np.array(rev), n_features=n_features)
    X_train.append(lst)
    y_train.append(rev)

for i in range(0, 200):
    lst = [random.randrange(0,9,1) for i in range(4)]
    rev = list(reversed(lst))
    lst = onehotencoder(lst, n_features=n_features)
    rev = onehotencoder(rev, n_features=n_features)
    X_test.append(lst)
    y_test.append(rev)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

def model_enc_dec(batch_size):
    encoder_inputs = Input(shape=(4, n_features), name='encoder_inputs')
    encoder_lstm = LSTM(n_units, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    states = [state_h, state_c]

    decoder_inputs = Input(shape=(1, n_features), name='decoder_inputs')
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_dense = Dense(n_features, activation='softmax', name='decoder_dense')

    all_outputs = []

    decoder_input_data = np.zeros((batch_size, 1, n_features))
    decoder_input_data[:, 0, 0] = 1

    inputs = decoder_input_data

    for i in range(4):
        outputs, state_h, state_c = decoder_lstm(inputs, initial_state=states)
        outputs = decoder_dense(outputs)
        all_outputs.append(outputs)

        inputs = outputs
        states = [state_h, state_c]

    decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

    model = Model(encoder_inputs, decoder_outputs, name='model_encoder_decoder')
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


batch_size = 10
model = model_enc_dec(batch_size=batch_size)
#print(model.summary())
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=500, validation_split=0.1, callbacks=[es])

# evaluate the model
_, train_acc = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
print('\nPREDICTION ACCURACY (%):')
print('Train: %.3f, Test: %.3f' % (train_acc*100, test_acc*100))
