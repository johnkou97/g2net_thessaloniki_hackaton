import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def npz_to_tensor(loc:str):
    data = np.load(loc)
    
    array1 = data['foreground']
    tensor1 = tf.convert_to_tensor(array1)
    
    if 'label' in data:
        array2 = data['label']
        tensor2 = tf.convert_to_tensor(array2,dtype=tf.int32)
        return tensor1, tensor2
    
    return tensor1

import matplotlib.pyplot as plt

def plot_loss_and_accuracy(history,name: str = 'loss_accuracy'):
    # Get training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get training and validation accuracy
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(loss) + 1), loss, label='Training')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(accuracy) + 1), accuracy, label='Training')
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    plt.close()
