import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm  # sklearn 自带 SVM 分类器

X = [[1, 1],
     [2, 2],
     [2, 0],
     [0, 0],
     [1, 0],
     [0, 1]]
X = np.array(X)

Y = [0, 0, 0, 1, 1, 1]
Y = np.array(Y)




clf = svm.SVC(kernel='linear', C=100000000)  # 创建线性 SVM 分类器
clf.fit(X, Y)  # 拟合数据

# 绘制离散数据点
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, s=30)

# 绘制决策函数
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格来评估模型
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
XX, YY = np.meshgrid(xx, yy)

xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# 绘制决策边界和边距
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# 绘制支持向量（Support Vectors）
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none',
           edgecolors='k')

plt.show()  # 显示