# Import libraries

import numpy as np 
from sklearn import preprocessing

# Extract data from CSV

raw_cvs_data = np.loadtxt('audiobooks.csv', delimiter=',')

unscaled_inputs_all = raw_cvs_data[:,1:-1]
targets_all = raw_cvs_data[:,-1]

# Balance the dataset

num_one_targets = int(np.sum(targets_all))
zero_targets_counter = 0
indices_to_remove = []

for i in range(targets_all.shape[0]):
    if targets_all[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_inputs_equal_priors = np.delete(unscaled_inputs_all, indices_to_remove, axis = 0)
targets_equal_priors = np.delete(targets_all, indices_to_remove, axis = 0)

# Standardize the inputs

scaled_inputs = preprocessing.scale(unscaled_inputs_equal_priors)

# Shuffle the data

shuffled_indices = np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs = scaled_inputs[shuffled_indices]
shuffled_targets = targets_equal_priors[shuffled_indices]

# Split the dataset into train, validation and test 

samples_count = shuffled_inputs.shape[0]

train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count:]

# Save to *.npz

np.savez('audiobooks_train', inputs=train_inputs, targets=train_targets)
np.savez('audiobooks_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('audiobooks_test', inputs=test_inputs, targets=test_targets)




