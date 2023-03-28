import h5py
import os

root = "./newdata"
master = "master.h5"

with h5py.File(master, 'a') as f:
    for file in os.listdir(root):
        print(len(f))
        with h5py.File(os.path.join(root,file), 'r') as f2:
            for key in f2.keys():
                f2.copy(key, f)

