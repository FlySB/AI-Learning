def loadDataSet(fileName):


    numFeat = len(open(fileName).readline().split('\t')) - 1

    dataMat = []
    labelData = []

    fr = open(fileName)

    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')

        for i in range(numFeat):
            lineArr.append(float(curLine[i]))

        dataMat.append(lineArr)
        labelData.append(float(curLine[-1]))
    return dataMat, labelData
