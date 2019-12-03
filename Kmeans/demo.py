from test import loadDataSet
from test import kMeans
from test import biKmeans
from test import clusterClubs
import numpy as np
s = "C:\\Users\\龚兴SUNNYMAN\\Desktop\\hw.txt"
imame = "C:\\Users\\龚兴SUNNYMAN\\Desktop\\xxx.png"

x = loadDataSet(s)
x = np.array(x)
y = [[9,3],[11,3]]
y =np.mat(y)
clusterClubs(s,imame,y,2)