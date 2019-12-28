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

def NbCount(mat):
    Nb = {}
    for i in range(len(mat)):
        list = []
        for j in range(len(mat)):
            if mat[i][j] == 1:
                list.append(j)
        Nb[i] = list
    return Nb

# 单次迭代
def SIRSpread_mat(mat, beta, mu, vec):
    """S:0 I:1 R:2"""
    for i in range(vec.size):
        if vec[i] == 0:
            num = 0
            for j in range(vec.size):
                if mat[i, j] == 1 and vec[j] == 1:
                    num = num + 1
            prob = 1 - (1 - beta) ** num
            rand = random.random()
            if rand < prob:
                vec[i] = 1
        elif vec[i] == 1:
            rand = random.random()
            if rand < mu:
                vec[i] = 2
    return vec

# 邻接表
def SIRSpread_Nb(Nbcount, beta, mu, vec):
    """S:0 I:1 R:2"""
    for i in range(vec.size):
        if vec[i] == 0:
            num = 0
            for v in Nbcount[i]:
                if vec[v] == 1:
                    num = num + 1
            prob = 1 - (1 - beta) ** num
            rand = random.random()
            if rand < prob:
                vec[i] = 1
        elif vec[i] == 1:
            rand = random.random()
            if rand < mu:
                vec[i] = 2
    return vec

# 判断网络中是否还有I点
def Find_I(vec):
    for i in vec:
        if i == 1:
            return True
    return False

# 计算图中R点
def Sum_R(vec):
    sumR = 0
    for i in vec:
        if i == 2:
            sumR += 1
    return sumR

# n次迭代求平均
def n_SIR_mat(mat, beta, mu, v, n):
    sum = 0
    for i in range(n):
        vec = np.zeros((1, mat.shape[0]))
        vec = np.array(vec[0])
        vec[v] = 1
        while Find_I(vec):
            SIRSpread_mat(mat, beta, mu, vec)
        sum += Sum_R(vec) #/len(vec)
    return sum/n

def score_SIR_mat(mat, beta, mu, n):
    lenth = mat.shape[0]
    score = {}
    for i in range(lenth):
        score[i] = n_SIR_mat(mat, beta, mu, i, n)
    return score

# n次迭代求平均
def n_SIR_Nb(Nb, beta, mu, v, n):
    sum = 0
    for i in range(n):
        vec = np.zeros((1, len(Nb)))
        vec = np.array(vec[0])
        vec[v] = 1
        while Find_I(vec):
            SIRSpread_Nb(Nb, beta, mu, vec)
        sum += Sum_R(vec) #/len(vec)
    return sum/n

def score_SIR_Nb(Nb, beta, mu, n):
    lenth = len(Nb)
    score = {}
    for i in range(lenth):
        score[i] = n_SIR_Nb(Nb, beta, mu, i, n)
    return score

def DegreeCount(mat, i):
    """mat: 邻接矩阵"""
    num = 0
    for j in range(len(mat)):
        if mat[i][j] == 1:
            num = num + 1
    return num

def DegreeRank_mat(mat):
    """mat: 邻接矩阵"""
    degree = {}
    for i in range(len(mat)):
        num = DegreeCount(mat,i)
        degree[i] =num
    return degree

def DegreeRank_Nb(Nb):
    degree = {}
    for k, v in Nb.items():
        degree[k] = len(v)
    return degree


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




def LocalRank(NbCount,DegreeCount):
    """
    :param NbCount: 网络邻接表
    :param DegreeCount: 节点度表
    :return: LocalRank
    """
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
    """
    :param ks: 网络邻接表
    :return: K-shell索引
    """
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
# 更适合于稀疏图
def Hindex(indexList):
    """
    :param indexList: (k1,k2,k3......kn)
    :return: H指数
    """
    indexSet = sorted(set(indexList), reverse=True)
    sign = 0
    for index in indexSet:
        clist = [i for i in indexList if i >= index]
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
#         # sum(CounterValues[0:index+1])
#         if CounterKeys[index] <= sum(CounterValues[0:index + 1]):
#             hinex = CounterKeys[index]
#             sign = 1
#             break
#     if sign == 0: hinex = len(indexList)
#     return hinex

# n趋近无穷大，H-index值不在变化
def n_Hindex(Nbcount,degree, n):
    """
    :param Nbcount: 网络邻接表
    :param degree: 节点度数表
    :param n: 迭代次数
    :return: H-index表
    """
    for i in range(1, n+1):
        h_NB = {}
        for k, v in Nbcount.items():
            list = []
            for i in v:
                list.append(degree[i])
            h_NB[k] = list
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
    # print(TheMax)
    List = []
    for k,v in TheDist.items():
        if v == TheMax: List.append(k)
    return List



def List_SIR(mat, beta, mu, vec_list, n):
    sum = 0
    for v in vec_list:
        sum += n_SIR_mat(mat,beta,mu,v,n)
    return sum/len(vec_list)

# 将k-shell转换成字典
def To_List(TheDist):
    lenth = 0
    first_dict = {}
    last_dict = {}
    for k, v in TheDist.items():
        lenth += len(v)
    for k, v in TheDist.items():
        for i in v:
            first_dict[i] = k
    for i in range(lenth):
        for k,v in first_dict.items():
            if k == i:
                last_dict[i] = v
    return last_dict

def Correlation(dist1, dist2):
    n1 = 0
    n2 = 0
    n = len(dist1)
    for i in range(n):
        for j in range(n):
            if (dist1[i] < dist1[j] and dist2[i] < dist2[j]) or (dist1[i] > dist1[j] and dist2[i] > dist2[j]):
                n1 += 1
            elif (dist1[i] == dist1[j] and dist2[i] == dist2[j]): n1 += 1
            elif (dist1[i] < dist1[j] and dist2[i] > dist2[j]) or (dist1[i] > dist1[j] and dist2[i] < dist2[j]):
                n2 += 1
    # print(n1)
    # print(n2)
    # print(n)
    t = (n1-n2)/(n*(n-1))
    return t




def NbCount_3(Nbcount):
    Nb = {}
    for k, v in Nbcount.items():
        list_3 = {}
        list_3[1] = v
        list_3[2] = []
        list_3[3] = []
        for i in v:
            for j in Nbcount[i]:
                if j != k and j not in list_3[1] and j not in list_3[2]: list_3[2].append(j)
        for i in list_3[2]:
            for j in Nbcount[i]:
                if j != k and j not in list_3[1] and j not in list_3[2] and j not in list_3[3]: list_3[3].append(j)
        Nb[k] = list_3
    return Nb

def score_Nb_3(Nb_3, Nb ,beta):
    score = {}
    for k, v in Nb_3.items():
        score_1 = {}
        for i in v[1]:
            score_1[i] = beta

        for i in v[2]:
            n = 1
            for j in Nb[i]:
                if j in v[1]: n = n*(1 - score_1[j]*beta)
            score_1[i] = 1 - n

        for i in v[3]:
            n = 1
            for j in Nb[i]:
                if j in v[2]: n = n*(1 - score_1[j]*beta)
            score_1[i] = 1 - n
        sum = 0
        for m, n in score_1.items():
            sum += n
        score[k] = sum
    return score


def All_data(data, beta):
    Score = {}
    for name in data:
        Score_list = {}
        path = "bigdata/" + name + ".txt"
        mat = CreateDataset(path)
        Nb = NbCount(mat)
        print(Nb)
        Nb_3 = NbCount_3(Nb)
        sir = score_Nb_3(Nb_3, Nb, beta)
        print(sir)
        degree = DegreeRank_Nb(Nb)
        print(degree)
        Score_list["DegreeRank"] = Correlation(sir, degree)
        L_R = LocalRank(Nb, degree)
        print(L_R)
        Score_list["LocalRank"] = Correlation(sir, L_R)
        H_I = n_Hindex(Nb, degree, 100)
        print(H_I)
        Score_list["H-index"] = Correlation(sir, H_I)
        ks = K_shell(Nb)
        k_s = To_List(ks)
        print(k_s)
        Score_list["K-shell"] = Correlation(sir, k_s)
        print(Score_list)
        Score[name] = Score_list

    return Score

def Change(score):
    s = {}
    txt= []
    a_s = []
    for k in score:
        txt.append(k)
    for k in score[txt[0]]:
        a_s.append(k)
    for i in range(len(score)):
        s[txt[i]] = []
        for k,v in score[txt[i]].items():
            s[txt[i]].append(v)
    return txt, a_s, s

import matplotlib.pyplot as plt
from pylab import *         #支持中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
def pante(S):
    data, a_s ,score = Change(S)
    marker = ['o','*','x','v','D','.','1','s']
    color = ['b','g','r','m','y','k','c']
    # pl.xlim(-1, 11) # 限定横轴的范围
    # pl.ylim(-1, 110) # 限定纵轴的范围
    # plt.plot(x, y, marker='o', color='r', label=u'y=x^2曲线图')
    # plt.plot(x, y1, marker='*', color='b', label=u'y=x^3曲线图')
    for i in range(len(data)):
        plt.plot(a_s, score[data[i]], marker=marker[i], color=color[i], label=str(data[i]))
    plt.legend()  # 让图例生效
    plt.xlabel(u"算法")  # X轴标签
    plt.ylabel(u"相关系数")  # Y轴标签
    plt.title(u"不同算法与概念模型对比")  # 标题

    plt.show()

txt = ["Grid","INT","NS","PB","PPI","USAir97"]

import time
start = time.time()
Score = All_data(txt, 0.6)
print(Score)
data, a,s = Change(Score)
print(data)
print(a)
print(s)
pante(Score)
end = time.time()
print(end - start)


# m = CreateDataset("bigdata/"+txt[4]+".txt")
# import time
# start = time.time()
# sir = score_SIR(m, 0.6, 0.8, 1)
# print(sir)
# end = time.time()
# print(end-start)

# Nb = NbCount(m)
# print(Nb)
#
# Nb_3 = NbCount_3(Nb)
# print(Nb_3)
#
# score = score_Nb_3(Nb_3, Nb, 0.6)
# print(score)
#
# start = time.time()
# sir_1 = score_SIR_Nb(Nb, 0.6, 0.8, 1)
# print(sir_1)
# end = time.time()
# print(end - start)

#
# degree = DegreeRank(m)
# print(degree)
# print(Correlation(sir_1, degree))
# dr = Out(degree)
# print(dr)
# print(List_SIR(m,0.6,0.8,dr,100))
# print(n_SIR(m,0.6,0.8,12,10))
#
# v = np.zeros((1,len(degree)))
# v[0][8]=1
# v = np.array(v[0])
# SIR_Spread(Nb, 0.6,0.8,v)
# print(v)
# print(v[6])
#
# while Find_I(v):
#     SIRSpread(m, 0.6, 0.8, v)
#
# print(v)
# print(Find_I(v))
#
# print(Find_R(v))

# lR = LocalRank(Nb, degree)
# print(lR)
# print(Correlation(sir_1, lR))
# L_R = Out(lR)
# print(L_R)
# print(List_SIR(m,0.6,0.8,L_R,100))
# #
# # # K_shell使用后参数会变为空
#
#
#
# x = n_Hindex(Nb,degree,100)
# print(x)
# print(Correlation(sir_1, x))
# h_i = Out(x)
# print(h_i)
# print(List_SIR(m,0.6,0.8,h_i,100))


#
#
# ks = K_shell(Nb)
# k_s= To_List(ks)
# print(k_s)
# print(Correlation(sir_1, k_s))
#
# d_l = Correlation(lR,x)
# print(d_l)

# k_s = ks[len(ks)]
# print(k_s)
# print(List_SIR(m,0.6,0.8,k_s,100))

