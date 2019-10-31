import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn import svm, metrics, preprocessing, neighbors
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix
from mlxtend.data import loadlocal_mnist



train_X, train_y = loadlocal_mnist(
    images_path='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/train-images-idx3-ubyte',
    labels_path='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/train-labels-idx1-ubyte')

test_X, test_y = loadlocal_mnist(
    images_path='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/t10k-images-idx3-ubyte',
    labels_path='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/t10k-labels-idx1-ubyte')

if not os.path.isfile('C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/train-images.csv'):
    # Training images and labels
    np.savetxt(fname='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/train-images.csv',
               X=train_X, delimiter=',', fmt='%d')
    np.savetxt(fname='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/train-labels.csv',
               X=train_y, delimiter=',', fmt='%d')
    # Test images and labels
    np.savetxt(fname='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/test-images.csv',
               X=test_X, delimiter=',', fmt='%d')
    np.savetxt(fname='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/test-labels.csv',
               X=test_y, delimiter=',', fmt='%d')

# X_train = pd.read_csv('C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/train-images.csv', sep=',')
# y_train = pd.read_csv('C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/train-labels.csv', sep=',')
X_test = pd.read_csv('C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/test-images.csv', sep=',')
y_test = pd.read_csv('C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/test-labels.csv', sep=',')

# Scale data
print("Scaling data")
X_train = preprocessing.scale(X_test)
X_test = preprocessing.scale(X_test)
# Split test set into training and test.
X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size = .33)

#
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

print(clf)

y_expect = y_test
y_pred = clf.predict(X_test)

print(metrics.classification_report(y_expect, y_pred))

cm = ConfusionMatrix(clf, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
cm.fit(X_train, y_train)
cm.score(X_test, y_test)

cm.show()

