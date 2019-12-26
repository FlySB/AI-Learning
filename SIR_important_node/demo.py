import numpy as np
import random
from collections import Counter

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

def Find_I(vec):
    for i in vec:
        if i == 1:
            return True
    return False

def Find_R(vec):
    sumR = 0
    for i in vec:
        if i == 2:
            sumR += 1
    return sumR

def DegreeCount(mat, i):
    num = 0
    for j in range(len(mat)):
        if mat[i][j] == 1:
            num = num + 1
    return num

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

def DegreeRank(mat):
    degree = {}
    for i in range(len(mat)):
        num = DegreeCount(mat,i)
        degree[i] =num
    return degree


def LocalRank(NbCount,DegreeCount):

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

# H-index指数
def Hindex(indexList):
    indexSet = sorted(set(indexList), reverse=True)
    sign = 0
    for index in indexSet:
        # clist为大于等于指定引用次数index的文章列表
        clist = [i for i in indexList if i >= index]
        # 由于引用次数index逆序排列，当index<=文章数量len(clist)时，得到H指数
        if index <= len(clist):
            sign = 1
            break
    if sign == 0: index = len(indexList)
    return index

# # 更适合稠密图
# def Hindex(indexList):
#     sign = 0
#     HCounter = Counter(indexList)
#     ReversedCounter = sorted(HCounter.items(), reverse=True)
#     CounterKeys = [i[0] for i in ReversedCounter]
#     CounterValues = [i[1] for i in ReversedCounter]
#     for index in range(0, len(CounterValues)):
#         # sum(CounterValues[0:index+1])为大于等于某个索引值——CounterKeys[index]的所有的文章总和
#         if CounterKeys[index] <= sum(CounterValues[0:index + 1]):
#             hinex = CounterKeys[index]
#             sign = 1
#             break
#     if sign == 0: hinex = len(indexList)
#     return hinex

def n_Hindex(Nbcount,degree, n):
    for i in range(1, n+1):
        h_NB = {}
        for k, v in Nbcount.items():
            list = []
            for i in v:
                list.append(degree[i])
            h_NB[k] = list
        # print("Nbcount")
        # print(Nbcount)
        # print("degree")
        # print(degree[975])
        # print(degree)
        # print("h_NB")
        # print(h_NB)
        h = {}
        for k, v in h_NB.items():
            h[k] = Hindex(v)
        # k1, k2 , k3~~~更新一遍
        degree = h

    return h

def Out(TheDist):
    TheMax = 0
    for k,v in TheDist.items():
        if v > TheMax: TheMax = v
    print(TheMax)
    List = []
    for k,v in TheDist.items():
        if v == TheMax: List.append(k)
    return List


m = CreateDataset("/Users/gong/Desktop/powernet.txt")

Nb = NbCount(m)

degree = DegreeRank(m)
print(degree)
print(Out(degree))

v = np.zeros((1,len(degree)))
v[0][2553]=1
v = np.array(v[0])
print(v)
print(v[2553])

while Find_I(v):
    v = SIRSpread(m, 0.6, 0.8, v)

print(v)
print(Find_I(v))

print(Find_R(v))

# lR = LocalRank(Nb, degree)
# print(lR)
# print(Out(lR))
# #
# # # K_shell使用后参数会变为空
#
#
#
x = n_Hindex(Nb,degree,100)
print(x)
print(Out(x))

v = np.zeros((1,len(degree)))
v[0][4332]=1
v = np.array(v[0])
print(v)
print(v[4332])

while Find_I(v):
    v = SIRSpread(m, 0.6, 0.8, v)

print(v)
print(Find_I(v))

print(Find_R(v))

#
#
# ks = K_shell(Nb)
# print(ks[len(ks)])