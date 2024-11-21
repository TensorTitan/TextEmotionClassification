from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


def create_ANN(input_dim):
    model = Sequential()

    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(6, activation='softmax'))

    return model