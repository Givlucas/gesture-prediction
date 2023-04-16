import multiprocessing as mp
import os
import scipy.io
import h5py
import matplotlib.pyplot as plt
import numpy as np


def RMS(frame):
    summ = np.zeros((1, 12))
    for x in frame:
        x = np.array(x)
        summ += x
    summ = summ**2
    return np.sqrt(summ)


def digest(i_path, o_path, num_frame, frame_size, gain, name):
    '''
    # algorith:
    # For each database and for each file in the data base and for each sensor.
    # Take an RMS over the number of samples needed to downsample to 200hz and apply gain of 150ms
    # store in array where first variable is exersise number. 
    # once the number of RMS stored is greater
    # than 150ms of time then append to list
    '''
    summ = 0
    mat = scipy.io.loadmat(i_path)
    with h5py.File(o_path, 'a') as f:
        # get length of set
        length = len(mat['emg'])
        # create buf to store new sets until there is enough
        buf = []
        for i in range(length):
            # Calculate RMS for frame from i to I + frame size
            if i + frame_size > len(mat['emg']):
                break
            # collect data for frame
            frame = []
            for x in range(frame_size):
                if mat['stimulus'][i + x] == 0:
                    frame = []
                else:
                    exersise = mat['stimulus'][i + x]
                    frame.append(mat['emg'][i + x])

            if len(frame) == 0:
                continue
            else:
                frame = np.array(frame)
            buf.append(RMS(frame) * gain)
            if len(buf) >= num_frame:  # Check to see if there is enough to create a group
                group = f.create_group(f'{name}_data_{summ}')
                summ += 1
                group.attrs['label'] = exersise
                group.create_dataset('data', data=buf)
                buf = []


if __name__ == '__main__':

    # Sampling rate of data
    hz = 2000
    # Size of moving rms window
    frame_size = 1801
    # Number of frames = 150ms of time
    num_frame = (((hz - frame_size) + 1) // (1 / 0.150)) + 1
    # set the gain
    gain = 14000
    # where databases are stored
    root = "./old-data"
    # where new data files are too be stored
    new = "./data"
    # List to store paths of all data files
    file_list = []
    # Iterate through all file in each data base
    summ = 0
    for dir in os.listdir(root):
        database = os.path.join(root, dir)
        if os.path.isdir(database):
            # Iterate through each sample
            for sample in os.listdir(database):
                sample = os.path.join(database, sample)
                if os.path.isdir(sample):
                    # Iterate through each test in sample
                    for file in os.listdir(sample):
                        # Make sure file isnt hidden
                        if file[0] == '.':
                            continue
                        file = os.path.join(sample, file)
                        if os.path.isfile(file):
                            print(file)
                            file_list.append(file)

    i = 0
    arguments = []
    workers = 16
    # digest(file_list[50],os.path.join(new, "test"), num_frame, frame_size,gain, 1) 
    while i < len(file_list):
        arguments.append([file_list[i], os.path.join(
            new, f"s{i}"), num_frame, frame_size, gain, f"s{i}"])
        i += 1
    with mp.Pool(processes=workers) as pool:
        pool.starmap(digest, arguments)
