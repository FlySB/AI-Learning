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
        Mat[int(Arr[1])][int(Arr[0])] = 1
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


def DegreeCount(mat, i):
    num = 0
    for j in range(len(mat)):
        if mat[i][j] == 1:
            num = num + 1
    return num

def DegreeRank(mat):
    degree = {}
    for i in range(len(mat)):
        num = DegreeCount(mat,i)
        degree[i] =num
    return degree

def NbCount(mat):
    Nb = {}
    for i in range(len(mat)):
        list = []
        for j in range(len(mat)):
            if mat[i][j] == 1:
                list.append(j)
        Nb[i] = list
    return Nb
#
# def QCount(DegreeCount, NbCount):
#     Q = {}
#     for i in range(len(NbCount)):
#         num = 0
#         for nb in NbCount[i]:
#             num = num + DegreeCount[nb]
#         Q[i] = num
#     return Q
#
# def CLCount(NbCount,Qcount):
#     CL = {}
#     for i in range(len(NbCount)):
#         num = 0
#         for nb in NbCount[i]:
#             num = num + Qcount[nb]
#         CL[i] = num
#     return CL

def LocalRank(mat,DegreeCount):
    NbCount = {}
    for i in range(len(mat)):
        list = []
        for j in range(len(mat)):
            if mat[i][j] == 1:
                list.append(j)
        NbCount[i] = list

    Qcount = {}
    for i in range(len(NbCount)):
        num = 0
        for nb in NbCount[i]:
            num = num + DegreeCount[nb]
        Qcount[i] = num

    CL = {}
    for i in range(len(NbCount)):
        num = 0
        for nb in NbCount[i]:
            num = num + Qcount[nb]
        CL[i] = num
    return CL

def K_shell(ks):
    kshell = {}
    ks1 = []
    # k值的初始值为s=1
    s = 1
    kshell[s] = []
    while ks:
        # 根据度数值是否和s相等，将找到的节点存入ks1
        for k, v in ks.items():
            if len(v) <= s:
                ks1.append(k)
        for i in ks1:
            kshell[s].append(i)
        # 判断是否ks1是否含有值，如果含有即代表s不用增加，否则是s+1
        if ks1 != []:
            for k2 in ks1:
                # 如果记录的节点在网络中，则删除
                if k2 in ks:
                    ks.pop(k2)
                # 删除的那个节点不应该出现在网络节点的度数中，所以我们要移除
                for v in ks.values():
                    if k2 in v:
                        v.remove(k2)
        else:
            s += 1
            kshell[s] = []
        # 每一次进行完上面的代码都要进行一次清空 ，为下次判断是否为None做好条件

        ks1.clear()
    return kshell




m = CreateDataset("/Users/gong/Desktop/text.txt")
# for i in range(len(m)):
#     for j in range(len(m)):
#         if m[i][j] == 1:
#             print(str(i)+"\t"+str(j))
degree = DegreeRank(m)
print(degree)

Nb = NbCount(m)
print(Nb)
ks = K_shell(Nb)
print(ks)

lR = LocalRank(m, degree)
print(lR)