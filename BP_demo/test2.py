import numpy as np
temp = [[1, 2, 3],[4, 5, 6],[7, 8, 9]]

x = np.array(temp)

y = np.ones((x.shape[0], x.shape[1]+1))
p =x[:, :2]
print(p)