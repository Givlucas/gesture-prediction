import tensorflow as tf
from tensorflow import keras
import os
import h5py as h5
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def load_data(file_path):
    # file_path = file_path.numpy()
    with h5.File(file_path, 'r') as f:
        for set in f.keys():
            window = [x for x in f[set]['data']]
            label = f[set].attrs['label']
            yield tf.reshape(window, (30, 12, 1)), label


if __name__ == '__main__':
    data_root = "./data"
    file_paths = [os.path.join(data_root, dir) for dir in os.listdir(data_root)]

    print(file_paths)
    dataset = tf.data.Dataset.list_files('./data/*')
    dataset = dataset.interleave(lambda x: tf.data.Dataset.from_generator(
        load_data,
        (tf.float32, tf.float32),
        ((30, 12, 1), (50, )),
        [x]), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=10000)

    # Batch the data
    batch_size = 256
    dataset = dataset.batch(batch_size)

    # Prefetch the data
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=12, strides=1,  activation='relu', input_shape=(30, 12, 1), padding='same'),  #C1
        keras.layers.Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='valid'),  #C3
        keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1),  #S4
        keras.layers.Conv2D(64, kernel_size=5, strides=1, activation='relu', padding='valid'),  #C5
        keras.layers.AveragePooling2D(pool_size=(3, 3), strides=1),  #S4
        keras.layers.Conv2D(64, kernel_size=(9, 1), strides=1, activation='relu', padding='valid'),  #C5
        keras.layers.Conv2D(64, kernel_size=1, strides=1, activation='relu', padding='valid'),  #C5
        keras.layers.Flatten(),  #Flatten
        keras.layers.Dense(84, activation='relu'),  #F6
        keras.layers.Dense(50, activation='softmax')  #Output layer
    ])

    sgdm = keras.optimizers.SGD(0.001, 0.9, weight_decay=0.0005)
    model.compile(optimizer=sgdm, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    model.fit(dataset, epochs=30)
