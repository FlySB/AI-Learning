# import scipy.integrate as spi
# import numpy as np
# import pylab as pl
#
# beta = 1.4247
# gamma = 0.14286
# TS = 1.0
# ND = 70.0
# S0 = 1 - 1e-6
# I0 = 1e-6
# INPUT = (S0, I0, 0.0)
#
#
# def diff_eqs(INP, t):
#     '''The main set of equations'''
#     Y = np.zeros((3))
#     V = INP
#     Y[0] = - beta * V[0] * V[1]
#     Y[1] = beta * V[0] * V[1] - gamma * V[1]
#     Y[2] = gamma * V[1]
#     return Y  # For odeint
#
#
# t_start = 0.0;
# t_end = ND;
# t_inc = TS
# t_range = np.arange(t_start, t_end + t_inc, t_inc)
# print(t_range)
# RES = spi.odeint(diff_eqs, INPUT, t_range)

#print(RES)

# ks代表了一个简单的网络，里面由三个节点‘1’，‘2’，‘4’
ks = {1:[2,3,4,5],2:[1,4],3:[1],4:[1,2,5],5:[1,4]}
#ks1暂时存储我们的查询结果
kshell = {}
ks1 = []
    # k值的初始值为s=1
s = 1
kshell[s] = []
while ks:
    #根据度数值是否和s相等，将找到的节点存入ks1
    for k,v in ks.items():
        if len(v) <= s:
            ks1.append(k)
    for i in ks1:
        kshell[s].append(i)
    #判断是否ks1是否含有值，如果含有即代表s不用增加，否则是s+1
    if ks1!=[]:
        for k2 in ks1:
            #如果记录的节点在网络中，则删除
            if k2 in ks:
                ks.pop(k2)
            #删除的那个节点不应该出现在网络节点的度数中，所以我们要移除
            for v in ks.values():
                if k2 in v:
                    v.remove(k2)
        print(ks)
    #     s += 1
    else:
        s += 1
        kshell[s] = []
    #每一次进行完上面的代码都要进行一次清空 ，为下次判断是否为None做好条件

    ks1.clear()
print(kshell)
