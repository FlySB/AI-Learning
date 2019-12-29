# import numpy as np
#
# # Save
# dictionary = {'hello':'world'}
# np.save('my_file.npy', dictionary)
#
# # Load
# read_dictionary = np.load('my_file.npy').item()
# print(read_dictionary['hello']) # displays "world"

import numpy as np
import time
import random
# from demo import *
# start = time.time()
# m = CreateDataset("bigdata/USAir97.txt")
# Nb = NbCount(m)
# sir  = score_SIR_Nb(Nb, 0.6, 0.8, 10)
# np.save('USAir97_sir_10.npy',sir)
# end = time.time()
# print(end-start)
# read = np.load('USAir97_sir_10.npy',allow_pickle=True)
# print(read)
# read = np.load('USAir97_sir_50.npy',allow_pickle=True)
# print(read)
# read = np.load('USAir97_sir_100.npy',allow_pickle=True)
# print(read)
# read = np.load('USAir97_sir_200.npy',allow_pickle=True)
# print(read)
# read = np.load('NS_sir_1.npy',allow_pickle=True)
# print(read)
# read = np.load('NS_sir_200.npy',allow_pickle=True) # 3061
# print(read)
read = np.load('PB_sir_1.npy',allow_pickle=True)
print(read)
read = np.load('PB_sir_200.npy',allow_pickle=True) # 3573
print(read)