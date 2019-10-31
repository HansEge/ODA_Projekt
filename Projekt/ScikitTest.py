import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import svm, metrics, preprocessing, neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ConfusionMatrix
from mlxtend.data import loadlocal_mnist


def idx2csv(X, y, name):
    if not os.path.isfile('C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/X_' + name + '.csv'):
        # Training images and labels
        np.savetxt(fname='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/X_' + name + '.csv',
                   X=X, delimiter=',', fmt='%d')
        np.savetxt(fname='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/y_' + name + '.csv',
                   X=y, delimiter=',', fmt='%d')

        return


def load_csv():
    X_train = pd.read_csv('C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/train-images.csv', sep=',')
    y_train = pd.read_csv('C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/train-labels.csv', sep=',')
    X_test = pd.read_csv('C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/test-images.csv', sep=',')
    y_test = pd.read_csv('C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/test-labels.csv', sep=',')

    return [X_train, y_train, X_test, y_test]


def load_idx():
    X_train, y_train = loadlocal_mnist(
        images_path='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/train-images-idx3-ubyte',
        labels_path='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/train-labels-idx1-ubyte')

    X_test, y_test = loadlocal_mnist(
        images_path='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/t10k-images-idx3-ubyte',
        labels_path='C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/t10k-labels-idx1-ubyte')
    return (X_train, y_train, X_test, y_test)


def make_confusion_matrix(model, X_tra, y_tra, X_te, y_te):
    clas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cm = ConfusionMatrix(model, classes=clas)
    cm.fit(X_tra, y_tra)
    cm.score(X_te, y_te)

    cm.show()

    return


def scale_mnist(Xtrain, Xtest):
    Xtrain = StandardScaler().fit_transform(Xtrain)
    Xtest = StandardScaler().fit_transform(Xtest)
    return Xtrain, Xtest

# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# Perform PCA to images and concat with labels
def performPCA(X, y, dim):
    pca = PCA(n_components=dim)

    principalComponents = pca.fit_transform(X)

    if dim == 2:
        principalDf = pd.DataFrame(data=principalComponents, columns=['Principle component 1', 'Principle component 2'])
    if dim == 3:
        principalDf = pd.DataFrame(data=principalComponents, columns=['Principle component 1', 'Principle component 2', 'Principle component 3'])

    labelsDf = pd.DataFrame(data=y, columns=['Numbers'])
    finalDf = pd.concat([principalDf, labelsDf[['Numbers']]], axis = 1)

    return finalDf

# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# Make plot of numbers with two principle components (2D)
def plt_pca(pca_df):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '0.50', '0.25', '0.75']
    for numbers, color in zip(targets, colors):
        indicesToKeep = pca_df['Numbers'] == numbers
        ax.scatter(pca_df.loc[indicesToKeep, 'Principle component 1']
                   , pca_df.loc[indicesToKeep, 'Principle component 2']
                   , c=color
                   , s=10)
    ax.legend(targets)
    ax.grid()
    return

# 3D scatterplot for the lulz
def plt_pca_3d(pca_df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_zlabel('Principal Component 3', fontsize=15)
    ax.set_title('3 component PCA', fontsize=20)

    targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '0.50', '0.25', '0.75']
    for numbers, color in zip(targets, colors):
        indicesToKeep = pca_df['Numbers'] == numbers
        ax.scatter(pca_df.loc[indicesToKeep, 'Principle component 1']
                   , pca_df.loc[indicesToKeep, 'Principle component 2']
                   , pca_df.loc[indicesToKeep, 'Principle component 3']
                   , c=color
                   , s=10)
    ax.legend(targets)
    ax.grid()

    return


if __name__ == '__main__':
    # Load MNIST data
    X_train, y_train, X_test, y_test = load_idx()

    # Scale data
    print("Scaling data")
    # X_train, X_test= scale_mnist(X_train, X_test)

    print("Perform PCA")
    pca_df = performPCA(X_train,y_train,3)
    plt_pca_3d(pca_df)



'''

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)

    print(clf)

    y_expect = y_test
    y_pred = clf.predict(X_test)

    print(metrics.classification_report(y_expect, y_pred))

    make_confusion_matrix(clf, X_train, y_train, X_test, y_test)
'''