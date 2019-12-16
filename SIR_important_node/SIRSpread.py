import numpy as np
import matplotlib.pyplot as plt
import random

def CreateER(N, p):
    mat = np.random.rand(N, N)
    mat = np.where(mat>p, 0, 1)
    for i in range(N):
        mat[i, i] = 0
        mat[i, :] = mat[:, i]
    return mat

def Distribution(mat):
    (a, b) = mat.shape
    Count = np.array([mat[i, :].sum() for i in range(a)])
    hist = np.histogram(Count, bins=1000, range=(0,1000))
    plt.plot(hist[0])
    plt.xlabel('degree')
    plt.ylabel('p(degree)')
    plt.show()
    return hist


# 对一个mat进行一次SIR的传播 S 1 -- I 2 -- R 3 普通人--1 感染者--2 恢复者
def SIRSpread(mat, beta, mu, vec):
    nvec = np.array(vec)
    for i in range(vec.size):
        if vec[i] == 1:
            num = 0
            for j in range(vec.size):
                if mat[i, j] == 1 and vec[j] == 2:
                    num = num + 1
            prob = 1 - (1 - beta) ** num
            rand = random.random()
            if rand < prob:
                nvec[i] = 2
        elif vec[i] == 2:
            rand = random.random()
            if rand < mu:
                nvec[i] = 3
    return nvec


# 设置传播次数，来进行传播，并返回每个阶段S，I，R的数量
def MultiSpread(N, beta, mu, t):
    mat = CreateER(N, 0.01)
    print("MAT")
    print(mat)
    vec = np.array([1 for i in range(N)])
    print("VEC")
    print(vec)

    rNum = random.randint(0, N - 1)
    vec[rNum] = 2

    S = []
    I = []
    R = []

    for i in range(t):
        vec = SIRSpread(mat, beta, mu, vec)
        print(vec)
        S.append(np.where(np.array(vec) == 1, 1, 0).sum())
        I.append(np.where(np.array(vec) == 2, 1, 0).sum())
        R.append(np.where(np.array(vec) == 3, 1, 0).sum())

    return S, I, R

# mat = CreateER(5, 0.01)
# print("MAT")
# print(mat)
# vec = np.array([1 for i in range(5)])
# print("VEC")
# print(vec)

S, I, R = MultiSpread(100,0.5,0.8,2)
print(S)
print(I)
print(R)