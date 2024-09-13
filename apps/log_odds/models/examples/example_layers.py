from tensorflow.keras.layers import Dense

LAYERS = {
    "D8": lambda: Dense(8, activation='relu'),
    "D16": lambda: Dense(16, activation='relu'),
    "D32": lambda: Dense(32, activation='relu'),
}