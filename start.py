# Import libraries 

import numpy as np 
import tensorflow as tf

# Load data 

npz = np.load('audiobooks_train.npz')
train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)

npz = np.load('audiobooks_validation.npz')
validation_inputs = npz['inputs'].astype(np.float)
validation_targets = npz['targets'].astype(np.int)

npz = np.load('audiobooks_test.npz')
test_inputs = npz['inputs'].astype(np.float)
test_targets = npz['targets'].astype(np.int)

# Create model 

# input_size = 10
output_size = 2
hidden_layer_size = 50

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

batch_size = 100 
max_epochs = 100
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

model.fit(
    train_inputs, 
    train_targets, 
    batch_size = batch_size,
    epochs = max_epochs,
    callbacks=[early_stopping],
    validation_data = (validation_inputs, validation_targets),
    verbose = 2
)

# Test the model 

test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)



