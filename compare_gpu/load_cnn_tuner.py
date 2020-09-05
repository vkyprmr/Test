'''
Developer: vkyprmr
Filename: load_cnn_tuner.py
Created on: 2020-06-29 at 13:30:28
'''
'''
Last modified on: 2020-06-29 at 14:10:58
'''


#%%
# Imports
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import argparse

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import TensorBoard

#%%
# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-gm", "--gpumemory",dest = "gpu_memory", help="GPU Memory to use - default 2 GB")
parser.add_argument("-m", "--mode",dest = "mode", help="Mode: 'static':'s' or 'dynamic':'d'")
parser.add_argument("-bs", "--batchsize",dest = "batch_size", help="Batch size")
parser.add_argument("-e", "--epochs",dest = "epochs", help="Epochs")
parser.add_argument("-d", "--device",dest = "device", help="CPU/GPU - default: GPU")


args = parser.parse_args()
try:
    batch_size = int(args.batch_size)
except:
    batch_size = 64
print(batch_size)
try:
    mode = args.mode.lower()
except:
    mode = 'd'
print(mode)
try:
    epochs = int(args.epochs)
except:
    epochs = 10
print(epochs)
try:
    device = args.device.lower()
except:
    device = 'no_device'
print(device)
try:
    gpu_mem = int(args.gpu_memory)
except:
    gpu_mem = 2
print(gpu_mem)
###

""" 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = InteractiveSession(config=config)
 """

if device=='gpu':
    if mode=='s' or mode=='static':
        #gpu_mem = int(args.gpu_memory)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        #The variable GB is the memory size you want to use.
        try:
            config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*gpu_mem))]
            if gpus:
                # Restrict TensorFlow to only allocate 1*X GB of memory on the first GPU
                try:
                    tf.config.experimental.set_virtual_device_configuration(gpus[0], config)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
                except RuntimeError as e:
                    # Virtual devices must be set before GPUs have been initialized
                    print(e)
        except:
            print('Static mode selected but no memory limit set. Please set a memory limit by adding the flag -gm=X (gb) or --gpumemory=x (gb) after -m=s or --memory=s')
            quit()
    else:
        physical_devices = tf.config.experimental.list_physical_devices('GPU') 
        for physical_device in physical_devices: 
            tf.config.experimental.set_memory_growth(physical_device, True)

else:
    physical_devices = tf.config.experimental.list_physical_devices('CPU') 

#%%
# Load Data
#(train_imgs, train_labels), (test_imgs, test_labels) = datasets.cifar100.load_data()
(train_imgs, train_labels), (test_imgs, test_labels) = datasets.cifar10.load_data()


# Normalize pixel values to be between 0 and 1
'''
Normalizing is always recommended before working with any data
'''

train_imgs, test_imgs = train_imgs/255.0, test_imgs/255.0

#%%
# Class names

##CIFAR10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#%%
# Load tuner object
tuner = pickle.load(open("tuner_1576628824.pkl","rb"))
tuner.get_best_hyperparameters()[0].values
tuner.get_best_models()[0].summary()


""" 
# Convolutional Neural Network
cnn = Sequential(
    [
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3), padding='same'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10)
    ]
    )

 """

cnn = tuner.get_best_models()[0]
cnn.summary()

""" # Compile and train our model
cnn.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
log_dir = "logs\\fit\\" + str(batch_size) + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)
start = datetime.now()
history = cnn.fit(train_imgs, train_labels, epochs=epochs, batch_size=batch_size,
                  validation_data=(test_imgs, test_labels),
                  verbose=1, callbacks = [tensorboard_callback])
print(f'Time taken to train on 10 epochs: {datetime.now() - start}') """


#%%
# Plots
#try:
""" plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.legend(loc='lower right')

plt.plot(history.history['loss'], label='accuracy')
plt.plot(history.history['val_loss'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss and accuracy')
#except:
#    print(history.history.keys())


#%%
# Evaluation
test_loss, test_acc = cnn.evaluate(test_imgs,  test_labels, verbose=1)
print(test_acc)

#%%
# Plot
plt.show() """