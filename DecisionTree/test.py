from createTree import createTree
from createDataSet import createDataSet
import decisionTreePlot as dtPlot

dataSet, labels = createDataSet()
myTree = createTree(dataSet, labels)
dtPlot.createPlot(myTree)
