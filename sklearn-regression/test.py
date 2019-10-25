from sklearn import datasets
from sklearn.linear_model import LinearRegression
from scipy.stats import probplot
import matplotlib.pyplot as plt

boston = datasets.load_boston()

lr = LinearRegression()
lr.fit(boston.data, boston.target)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
predictions =lr.predict(boston.data)

f = plt.figure(figsize= (7,5))
ax = f.add_subplot(111)
probplot(boston.target - predictions, plot=ax)
plt.show()
