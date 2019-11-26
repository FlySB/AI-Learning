import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

train_data = [[0.697, 0.460, 0],
              [0.774, 0.376, 0],
              [0.634, 0.264, 0],
              [0.608, 0.318, 0],
              [0.556, 0.215, 0],
              [0.403, 0.237, 0],
              [0.666, 0.091, 1],
              [0.243, 0.267, 1],
              [0.245, 0.057, 1],
              [0.343, 0.099, 1],
              [0.639, 0.161, 1],
              [0.657, 0.198, 1]]
train_data = np.array(train_data)

test_data = [[0.481, 0.149, 0],
             [0.437, 0.211, 0],
             [0.360, 0.370, 1],
             [0.593, 0.042, 1],
             [0.719, 0.103, 1]]
test_data = np.array(test_data)

x1 = train_data[:,0:2]
y1 = train_data[:,2]
x2 = test_data[:,0:2]
y2 = test_data[:,2]

def Common(m1, m2):
    i = 0;
    size = np.size(m1)
    for n in range(size):
        if(m1[n] == m2[n]):
            i = i + 1
    return i

Cmat = ['1','100','10000','1000000','100000000']
Accuracy = []
for i in Cmat:
    c = int(i)
    base_svc = SVC(C=c, kernel='linear')
    # base_svc = SVC(C=c, kernel='rbf', gamma=0.5)
    base_svc.fit(x1, y1)
    preduictY = base_svc.predict(x2)
    sizeY = np.size(y2)
    num = Common(y2, preduictY)
    Accuracy.append(num/sizeY)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(Cmat,Accuracy,c = 'r', marker = '.')
plt.ylim(0,1)
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.set_title("Watermelon dataset 3.0a")
# ax.set_xlabel("density")
# ax.set_ylabel("Sugar content")
# for n in test_data:
#     x = n[0]
#     y = n[1]
#     z = n[2]
#     if(z == 0):
#         ax.scatter(x, y, c='r', marker='.')
#     else:ax.scatter(x,y, c='k', marker='.')
# plt.show()