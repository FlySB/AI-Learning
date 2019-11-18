def createDataSet():
    dataSet = [['green', 'collapse',  'clear', 'cloudy','yes'],
               ['black', 'collapse',  'clear', 'boring','yes'],
               ['black', 'collapse',  'clear', 'cloudy','yes'],
               ['green', 'slightly',  'clear', 'cloudy','yes'],
               ['black', 'slightly',  'slightly paste','cloudy', 'yes'],
               ['black', 'slightly',  'slightly paste','boring', 'no'],
               ['green', 'hard', 'clear', 'clear', 'no'],
               ['light white', 'slightly', 'slightly paste','boring',  'no'],
               ['black', 'slightly', 'clear', 'cloudy',  'no'],
               ['light white', 'collapse',  'fuzzy', 'cloudy','no'],
               ['green', 'collapse',  'slightly paste', 'boring','no']]
    labels = ['color', 'genti',  'texture','knock']
    return dataSet, labels