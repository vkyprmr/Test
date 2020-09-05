'''
Developer: vkyprmr
Filename: cnn_tuner.py
Created on: 13-06-2020 (Sat) at 23:56:17
'''
'''
Last modified on: 2020-06-29 at 14:11:04
'''



'''
The code is directly taken from: https://www.tensorflow.org/tutorials/images/cnn

Some minor changes might have been made.

'''
#%%
# Imports
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from tensorflow.keras.callbacks import TensorBoard

from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import pickle
import winsound

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
#print(batch_size)
try:
    mode = args.mode.lower()
except:
    mode = 'd'
#print(mode)
try:
    epochs = int(args.epochs)
except:
    epochs = 10
#print(epochs)
try:
    device = args.device.lower()
except:
    device = 'no_device'
#print(device)
try:
    gpu_mem = int(args.gpu_memory)
except:
    gpu_mem = 2
#print(gpu_mem)

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

'''
##CIFAR100
super_classes = ['acquatic mammals', 'fish', 'flowers', 'food containers', 'fruits and vegetables', 'household electrical devices', 'household furniture', 'insects', 'large carnivores', 'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees', 'vehicles 1', 'vehicles 2']

class_dict = {
    'acquatic mammals':['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium fish', 'flatfish', 'ray', 'shark', 'trout'], 
    'flowers': ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips'], 
    'food containers': ['bottles', 'bowls', 'cans', 'cups', 'plates'], 
    'fruits and vegetables': ['apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers'],
    'household electrical devices': ['clock', 'computer keyboard', 'lamp', 'telephone', 'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple', 'oak', 'palm', 'pine', 'willow'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup truck', 'train'],
    'vehicles 2': ['lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']
}

class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
'''

#%%
# Plot
'''
Credits: corochann
Taken from: https://corochann.com/cifar-10-cifar-100-dataset-introduction-1258.html
Modified: yes
'''
def plot_cifar(data, label, row, col, scale, label_list=None):
    fig_w = data[0][0].shape[0]/80*row*scale
    fig_h = data[0][0].shape[1]/80*row*scale
    fig, ax = plt.subplots(row, col, figsize=(fig_h, fig_w))
    for i in range(row * col):
        # train[i][0] is i-th image data with size 32x32
        image, label_index = data[i], label[i]
        image = image.transpose(0, 1, 2)
        r, c = divmod(i, col)
        ax[r][c].imshow(image)  # cmap='gray' is for black and white picture.
        if label_list is None:
            ax[r][c].set_title('label {}'.format(label_index))
        else:
            ax[r][c].set_title('{}: {}'.format(label_index, label_list[label_index[0]]))
        ax[r][c].axis('off')  # do not show axis value
    plt.tight_layout()   # automatic padding between subplots

#%matplotlib qt
#plot_cifar(train_imgs, train_labels, 5, 5, 25, label_list=class_names)

#%%
# Build model
def build_model(hp):  # random search passes this hyperparameter() object 
    model = Sequential()

    model.add(Conv2D(hp.Int('input_units',
                                min_value=32,
                                max_value=256,
                                step=32), (3, 3), input_shape=(32,32,3),))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(hp.Int('n_layers', 1, 4)):  # adding variation of layers.
        model.add(Conv2D(hp.Int(f'conv_{i}_units',
                                min_value=32,
                                max_value=256,
                                step=32), (3, 3)))
        model.add(Activation('relu'))

    model.add(Flatten()) 
    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    return model

LOG_DIR = "tuner\\logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=25,  # how many variations on model?
    executions_per_trial=5,  # how many trials per variation? (same model could perform differently)
    directory=LOG_DIR)

tuner.search_space_summary()

log_dir = "logs\\fit\\" + str(batch_size) + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)

start = datetime.now()
tuner.search(train_imgs, train_labels,
             epochs=epochs,
             batch_size=batch_size,
             callbacks=[tensorboard_callback],
             validation_data=(test_imgs, test_labels))
print(f'Time taken to complete {epochs} epochs: {datetime.now() - start}')

tuner.results_summary()


with open(f"tuner_{int(datetime.now())}.pkl", "wb") as f:
    pickle.dump(tuner, f)


""" tuner = pickle.load(open("tuner_1576628824.pkl","rb"))
tuner.get_best_hyperparameters()[0].values
tuner.get_best_models()[0].summary()
 """

'''
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

cnn.summary()

# Compile and train our model
cnn.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
log_dir = "logs\\fit\\" + str(batch_size) + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir, histogram_freq=1, profile_batch=0)
start = datetime.now()
history = cnn.fit(train_imgs, train_labels, epochs=epochs, batch_size=batch_size,
                  validation_data=(test_imgs, test_labels),
                  verbose=1, callbacks = [tensorboard_callback])
print(f'Time taken to train on 10 epochs: {datetime.now() - start}')
'''

winsound.MessageBeep()

""" 
#%%
# Plots
try:
    plt.plot(history.history['acc'], label='accuracy')
    plt.plot(history.history['val_acc'], label = 'val_accuracy')

    plt.plot(history.history['loss'], label='accuracy')
    plt.plot(history.history['val_loss'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss and accuracy')
    plt.legend()
except:
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

    plt.plot(history.history['loss'], label='accuracy')
    plt.plot(history.history['val_loss'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss and accuracy')
    plt.legend()


#%%
# Evaluation
test_loss, test_acc = cnn.evaluate(test_imgs,  test_labels, verbose=1)
print(test_acc)

#%%
# Plot
plt.show() """
