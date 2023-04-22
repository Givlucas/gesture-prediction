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
    batch_size = 32
    dataset = dataset.batch(batch_size)

    # Prefetch the data
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = keras.models.Sequential([
        keras.layers.Conv1D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=(30, 12), padding='same'),  #C1
        keras.layers.AveragePooling1D(),  #S2
        keras.layers.Conv1D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'),  #C3
        keras.layers.AveragePooling1D(),  #S4
        keras.layers.Conv1D(120, kernel_size=5, strides=1, activation='tanh', padding='valid'),  #C5
        keras.layers.Flatten(),  #Flatten
        keras.layers.Dense(84, activation='tanh'),  #F6
        keras.layers.Dense(50, activation='softmax')  #Output layer
    ])

    model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    model.summary()
    model.fit(dataset, epochs=10)
