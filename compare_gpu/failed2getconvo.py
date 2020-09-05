'''
Author: KubaMichalczyk (http://kubamichalczyk.github.io/)

'''


import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

#%%
parser = argparse.ArgumentParser()
parser.add_argument("-gm", "--gpumemory",dest = "gpu_memory", help="GPU Memory to use")
parser.add_argument("-m", "--mode",dest = "mode", help="Mode: 'static':'s' or 'dynamic':'d'")
parser.add_argument("-bs", "--batchsize",dest = "batch_size", help="Batch size")
parser.add_argument("-e", "--epochs",dest = "epochs", help="Epochs")


args = parser.parse_args()

batch_size = int(args.batch_size)
mode = args.mode.lower()
epochs = int(args.epochs)



#%%

if mode=='s' or mode=='static':
    gpu_mem = int(args.gpu_memory)
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


        


#%%
max_features = 2000
max_len = 500

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=max_features)

X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

model = tf.keras.models.Sequential()
model.add(layers.Embedding(max_features,
                           128,
                           input_length=max_len, 
                           name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

callbacks = [
    tf.keras.callbacks.TensorBoard(
        log_dir="my-log-dir",
        histogram_freq=1,
        embeddings_freq=1,
    )
]
#%%
history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    callbacks=callbacks)

# %%
