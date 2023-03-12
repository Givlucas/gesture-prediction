import os
import scipy.io
import h5py
import matplotlib.pyplot as plt
import numpy as np

# datafile = "./data/DB1/s1/S1_A1_E1.mat"
# mat = scipy.io.loadmat(datafiley)
# length = len(mat['emg'])
# time = np.array([i/100 for i in range(0, length, 1)])

# plt.plot(time, mat['emg'])
# plt.show()
# print(length)

def RMS(frame):
    sum = np.zeros((1,12))
    for x in frame:
        x = np.array(x)
        sum += x
    sum = sum**2
    return np.sqrt(sum)
        

# algorith:
# For each database and for each file in the data base and for each sensor.
# Take an RMS over 100 samples. and gain of 14k and subsample to 200hz. store in array where first
# variable is exersize number. once the number of RMS stored is greater
# then 150ms of time then append to list

# Sampling rate of data
hz = 2000
# Size of moving rms window
frame_size = 1801
# Number of frames = 150ms of time
num_frame = (((hz - frame_size) + 1) // (1/0.150)) + 1
# set the gain
gain = 14000
# where databases are stored
root = "./data"
# where new data files are too be stored
new = "./newdata"

sum = 0
# Iterate through all file in each data base
with h5py.File(os.path.join(new, "set.h5"), "a") as f:    
    sum = 0
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
                            
                            mat = scipy.io.loadmat(file)
                            
                            # Figure out how many frames will be in set
                            length = int(len(mat['emg']) // 2000 * num_frame)
                            # create buffer to store new sets until there is enough
                            buffer = []
                            for i in range(length):
                                # Calculate RMS for frame from i to I + frame size
                                buffer.append(RMS(mat['emg'][i : i + frame_size]) * gain)
                                if len(buffer) >= num_frame: # Check to see if there is enough if there is create group
                                    group = f.create_group(f'data_{sum}')
                                    sum += 1
                                    group.attrs['label'] = mat['exercise']
                                    group.create_dataset('data', data=buffer)
                                    buffer = []
                                    
                                
