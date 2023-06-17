import tensorflow as tf
from tensorflow.keras import models

from helper import npz_to_tensor

X_train, Y_train = npz_to_tensor('data/classification_training_data.npz')
X_val, Y_val = npz_to_tensor('data/classification_validation_data.npz')
X_test = npz_to_tensor('data/classification_test_data.npz')

#load model from model/ folder
model = models.load_model('model')

# Predict on all the sets
Y_train_pred = model.predict(X_train)
Y_val_pred = model.predict(X_val)
Y_test_pred = model.predict(X_test)

# check accuracy on all the sets
train_accuracy = tf.keras.metrics.sparse_categorical_accuracy(Y_train, Y_train_pred)
val_accuracy = tf.keras.metrics.sparse_categorical_accuracy(Y_val, Y_val_pred)

print('Train accuracy: ' + str(tf.reduce_mean(train_accuracy).numpy()))
print('Validation accuracy: ' + str(tf.reduce_mean(val_accuracy).numpy()))

# generate submission file
with open('submission.csv', 'w') as file:
    file.write('id,label\n')
    for i in range(len(Y_test_pred)):
        file.write(str(i) + ',' + str(tf.argmax(Y_test_pred[i]).numpy()) + '\n')
