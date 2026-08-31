import tensorflow as tf
from keras import layers, models
import keras

class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        # Define layers
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(10)
        
        # Define a trainable variable
        self.my_variable = self.add_weight(name='my_variable', 
                                           shape=(), 
                                           initializer='ones', 
                                           trainable=True)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        
        # Use the custom variable in some way
        x = x * self.my_variable
        return x

# Instantiate and compile the model
model = CustomModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Dummy data for example purposes
import numpy as np
x_train = np.random.random((100, 20))
y_train = np.random.randint(10, size=(100,))

# Train the model
model.fit(x_train, y_train, epochs=5)
dfsag = model.trainable_variables

gdfsfdg = model.my_variable
pass