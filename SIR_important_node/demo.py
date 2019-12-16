import numpy as np
import random

# def CreateDataset(file_path):
#     Zreo = 0
#     MaxNode = 0
#     fr = open(file_path)
#     for line in fr.readlines():
#         lineArr = line.strip().split()
#         if int(lineArr[0]) == 0 or int(lineArr[1]) == 0:
#             Zreo = 1
#         if int(lineArr[0]) > MaxNode:
#             MaxNode = int(lineArr[0])
#         if int(lineArr[1]) > MaxNode:
#             MaxNode = int(lineArr[1])
#     if Zreo == 0:
#         Mat = np.zeros((MaxNode,MaxNode))
#     else: Mat = np.zeros((MaxNode+1,MaxNode+1))
#     fr2 = open(file_path)
#     for line1 in fr2.readlines():
#         Arr = line1.strip().split()
#         if Zreo == 1:
#             Mat[int(Arr[0])][int(Arr[1])] = 1
#         else: Mat[int(Arr[0])-1][int(Arr[1])-1] = 1
#     return Mat

def CreateDataset(file_path):
    MaxNode = 0
    fr = open(file_path)
    for line in fr.readlines():
        lineArr = line.strip().split()
        if int(lineArr[0]) > MaxNode:
            MaxNode = int(lineArr[0])
        if int(lineArr[1]) > MaxNode:
            MaxNode = int(lineArr[1])
    Mat = np.zeros((MaxNode+1,MaxNode+1))
    fr2 = open(file_path)
    for line1 in fr2.readlines():
        Arr = line1.strip().split()
        Mat[int(Arr[0])][int(Arr[1])] = 1
    return Mat

def SIRSpread(mat, beta, mu, vec):
    nvec = np.array(vec)
    for i in range(vec.size):
        if vec[i] == 0:
            num = 0
            for j in range(vec.size):
                if mat[i, j] == 1 and vec[j] == 1:
                    num = num + 1
            prob = 1 - (1 - beta) ** num
            rand = random.random()
            if rand < prob:
                nvec[i] = 1
        elif vec[i] == 1:
            rand = random.random()
            if rand < mu:
                nvec[i] = 2
    return nvec

m = CreateDataset("C:\\Users\\NEWSUNNYMAN\\Desktop\\111.txt")
print(m)
