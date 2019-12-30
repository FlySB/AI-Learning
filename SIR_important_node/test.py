import numpy as np
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
# read = np.load('PB_sir_1.npy',allow_pickle=True)
# print(read)
# read = np.load('PB_sir_200.npy',allow_pickle=True) # 3573
# print(read)
# read = np.load('PPI_sir_200.npy',allow_pickle=True)
# data = read.item()
# print(data)
# print(type(data))
# def All_data_HI(data):
#     Score = {}
#     for name in data:
#         print("\n")
#         Score_list = {}
#         read = np.load(name + '_NB_3.npy', allow_pickle=True)
#         sir = read.item()
#         print(sir)
#         read = np.load(name + '_Hindex_1.npy', allow_pickle=True)
#         H_1 = read.item()
#         print(H_1)
#         Score_list["H_1"] = Correlation(sir, H_1)
#         # print(sir)
#         # np.save(name+'_NB_3.npy', sir)
#         read = np.load(name + '_Hindex_5.npy', allow_pickle=True)
#         H_5 = read.item()
#         print(H_5)
#         Score_list["H_5"] = Correlation(sir, H_5)
#         read = np.load(name + '_Hindex_10.npy', allow_pickle=True)
#         H_10 = read.item()
#         print(H_10)
#         Score_list["H_10"] = Correlation(sir, H_10)
#         read = np.load(name + '_Hindex_20.npy', allow_pickle=True)
#         H_20 = read.item()
#         print(H_20)
#         Score_list["H_20"] = Correlation(sir, H_20)
#         read = np.load(name + '_Hindex_50.npy', allow_pickle=True)
#         H_50 = read.item()
#         print(H_50)
#         Score_list["H_50"] = Correlation(sir, H_50)
#         read = np.load(name + '_Hindex_100.npy', allow_pickle=True)
#         H_100 = read.item()
#         print(H_100)
#         Score_list["H_100"] = Correlation(sir, H_100)
#         print(Score_list)
#         Score[name] = Score_list
#
#     return Score
#
# def All_data_HI_n(data,n):
#     Score = {}
#     for name in data:
#         print("\n")
#         Score_list = {}
#         read = np.load(name + '_NB_3.npy', allow_pickle=True)
#         sir = read.item()
#         sir = To_n_cell(sir, n)
#         print(sir)
#         read = np.load(name + '_Hindex_1.npy', allow_pickle=True)
#         H_1 = read.item()
#         H_1 = To_n_cell(H_1, n)
#         print(H_1)
#         Score_list["H_1"] = Correlation(sir, H_1)
#         # print(sir)
#         # np.save(name+'_NB_3.npy', sir)
#         read = np.load(name + '_Hindex_5.npy', allow_pickle=True)
#         H_5 = read.item()
#         H_5 = To_n_cell(H_5, n)
#         print(H_5)
#         Score_list["H_5"] = Correlation(sir, H_5)
#         read = np.load(name + '_Hindex_10.npy', allow_pickle=True)
#         H_10 = read.item()
#         H_10 = To_n_cell(H_10, n)
#         print(H_10)
#         Score_list["H_10"] = Correlation(sir, H_10)
#         read = np.load(name + '_Hindex_20.npy', allow_pickle=True)
#         H_20 = read.item()
#         H_20 = To_n_cell(H_20, n)
#         print(H_20)
#         Score_list["H_20"] = Correlation(sir, H_20)
#         read = np.load(name + '_Hindex_50.npy', allow_pickle=True)
#         H_50 = read.item()
#         H_50 = To_n_cell(H_50, n)
#         print(H_50)
#         Score_list["H_50"] = Correlation(sir, H_50)
#         read = np.load(name + '_Hindex_100.npy', allow_pickle=True)
#         H_100 = read.item()
#         H_100 = To_n_cell(H_100, n)
#         print(H_100)
#         Score_list["H_100"] = Correlation(sir, H_100)
#         print(Score_list)
#         Score[name] = Score_list
#
#     return Score
print("sir")
sir = np.load('sir_score.npy', allow_pickle=True)
print(sir)
print("sir_3")
sir3 = np.load('sir_score_3.npy', allow_pickle=True)
print(sir3)
print("sir_5")
sir5 = np.load('sir_score_5.npy', allow_pickle=True)
print(sir5)
print("sir_10")
sir10 = np.load('sir_score_10.npy', allow_pickle=True)
print(sir10)

print("\n\n")
print("NB")
nb = np.load('NB_score.npy', allow_pickle=True)
print(nb)
print("NB_3")
nb3 = np.load('NB_score_3.npy', allow_pickle=True)
print(nb3)
print("NB_5")
nb5 = np.load('NB_score_5.npy', allow_pickle=True)
print(nb5)
print("NB_10")
nb10 = np.load('NB_score_10.npy', allow_pickle=True)
print(nb10)