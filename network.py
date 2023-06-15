import tensorflow as tf
import json
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from helper import npz_to_tensor, plot_loss_and_accuracy

# read the data
X_train, Y_train = npz_to_tensor('data/classification_training_data.npz')
X_val, Y_val = npz_to_tensor('data/classification_validation_data.npz')
X_test = npz_to_tensor('data/classification_test_data.npz')


# define hyperparameters
with open('best_hyperparameters.json', 'r') as file:
    best_params = json.load(file)
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']
regularization_rate = best_params['regularization_rate']
dropout_rate = best_params['dropout_rate']
epochs = 100
patience = 5

# Define the model architecture
model = tf.keras.Sequential([
        layers.BatchNormalization(input_shape=(2048, 1)),
        layers.Conv1D(filters=4, kernel_size=64, kernel_regularizer=l2(regularization_rate)),
        layers.ELU(),
        layers.Conv1D(filters=4, kernel_size=32, kernel_regularizer=l2(regularization_rate)),
        layers.MaxPooling1D(pool_size=4),
        layers.ELU(),
        layers.Conv1D(filters=8, kernel_size=32, kernel_regularizer=l2(regularization_rate)),
        layers.ELU(),
        layers.Conv1D(filters=8, kernel_size=16, kernel_regularizer=l2(regularization_rate)),
        layers.MaxPooling1D(pool_size=3),
        layers.ELU(),
        layers.Conv1D(filters=16, kernel_size=16, kernel_regularizer=l2(regularization_rate)),
        layers.ELU(),
        layers.Conv1D(filters=16, kernel_size=16, kernel_regularizer=l2(regularization_rate)),
        layers.MaxPooling1D(pool_size=4),
        layers.ELU(),
        layers.Flatten(),
        layers.Dense(units=32, kernel_regularizer=l2(regularization_rate)),
        layers.Dropout(rate=dropout_rate),
        layers.ELU(),
        layers.Dense(units=16, kernel_regularizer=l2(regularization_rate)),
        layers.Dropout(rate=dropout_rate),
        layers.ELU(),
        layers.Dense(units=3, activation='softmax')
    ])


# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x=X_train,
			y=Y_train,
			epochs=epochs,
			verbose=1,
			batch_size=batch_size,
			validation_data=(X_val, Y_val),
            shuffle=True,
			callbacks=[
				EarlyStopping(monitor='val_loss', patience=patience),
				ReduceLROnPlateau(verbose=1, patience=patience, monitor='val_loss')
			])

# Plots	
plot_loss_and_accuracy(history)