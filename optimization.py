import optuna
import optuna.visualization as vis
import json
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from helper import npz_to_tensor, plot_loss_and_accuracy

# read the data
X_train, Y_train = npz_to_tensor('data/classification_training_data.npz')
X_val, Y_val = npz_to_tensor('data/classification_validation_data.npz')
X_test = npz_to_tensor('data/classification_test_data.npz')

# Define the objective function for Optuna
def objective(trial):
    # Define the search space for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.7)
    regularization_rate = trial.suggest_loguniform('regularization_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16, 32, 64])
    epochs = 40
    patience = 5

    # Define the model architecture using the hyperparameters
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

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model with the defined hyperparameters
    model.fit(x=X_train,
			y=Y_train,
			epochs=epochs,
			verbose=0,
			batch_size=batch_size,
			validation_data=(X_val, Y_val),
            shuffle=True,
			callbacks=[
				EarlyStopping(monitor='val_loss', patience=patience),
				ReduceLROnPlateau(verbose=0, patience=patience, monitor='val_loss'),
                ModelCheckpoint('best-weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)
			])
    model.load_weights('best-weights.h5')
    # Evaluate the model on the validation set
    loss, accuracy = model.evaluate(X_val, Y_val)

    return accuracy

# Create an Optuna study and optimize the hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters from the study
best_params = study.best_params

with open('best_hyperparameters.json', 'w') as file:
    json.dump(best_params, file)

# Plot the optimization history
fig = vis.plot_optimization_history(study)
fig.write_image('hyperparameter_plots/optimization_history.png', scale=3)

# Plot the parameter importances
fig = vis.plot_param_importances(study)
fig.write_image('hyperparameter_plots/parameter_importances.png', scale=3)

# Plot the parallel coordinate plot
fig = vis.plot_parallel_coordinate(study)
fig.write_image('hyperparameter_plots/parallel_coordinate_plot.png', scale=3)

# Plot the parameter importances
fig = vis.plot_edf(study)
fig.write_image('hyperparameter_plots/edf.png', scale=3)

# Plot the parallel coordinate plot
fig = vis.plot_slice(study)
fig.write_image('hyperparameter_plots/slice.png', scale=3)

