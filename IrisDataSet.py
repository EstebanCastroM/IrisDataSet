#import numpy, pandas, matplotlib and sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl

#import Iris dataset
from sklearn.datasets import load_iris

#load Iris dataset
iris = load_iris()

#show dimensions of the dataset
print(iris.data.shape)

#show the names of the four features
print(iris.feature_names)

#create X and y matrices with all features
X = iris.data[:, :]
Y = iris.target[:]

#Plot the first two features as a scatter plot
#no grid, labels at axes in spanish
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Sepal length vs. Sepal width')
plt.show()

#generame un histograma de las longitudes de los pétalos de las flores
plt.hist(X[:, 2])