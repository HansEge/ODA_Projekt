import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import neighbors

'''
# Used for ORL images
import cv2 as cv
from PIL import Image
'''

# Skal åbenbart bruges til 3D plot
from mpl_toolkits.mplot3d import Axes3D

from sklearn.pipeline import Pipeline
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
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


def load_idx(path):
    X_train, y_train = loadlocal_mnist(
        images_path=path + 'train-images-idx3-ubyte',
        labels_path=path + 'train-labels-idx1-ubyte')

    X_test, y_test = loadlocal_mnist(
        images_path=path + 't10k-images-idx3-ubyte',
        labels_path=path + 't10k-labels-idx1-ubyte')
    return (X_train, y_train, X_test, y_test)


def make_confusion_matrix(model, X_tra, y_tra, X_te, y_te):

    cm = ConfusionMatrix(model, classes=model.classes_)
    cm.fit(X_tra, np.ravel(y_tra, order = 'C'))
    cm.score(X_te, y_te)

    cm.show()

    return


def scale_mnist(Xtrain, Xtest):
    Xtrain = StandardScaler().fit_transform(Xtrain)
    Xtest = StandardScaler().fit_transform(Xtest)
    return Xtrain, Xtest


# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# Perform PCA to images and concat with labels
def perform_pca(X, y, dim):
    pca = PCA(n_components=dim)

    principalComponents = pca.fit_transform(X)

    if dim == 2:
        principalDf = pd.DataFrame(data=principalComponents, columns=['Principle component 1', 'Principle component 2'])

    if dim == 3:
        principalDf = pd.DataFrame(data=principalComponents, columns=['Principle component 1', 'Principle component 2',
                                                                      'Principle component 3'])

    else:
        print("Wrong dimensions in performPCA")
        return

    labelsDf = pd.DataFrame(data=y, columns=['Numbers'])
    finalDf = pd.concat([principalDf, labelsDf[['Numbers']]], axis=1)

    return finalDf


# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# Make plot of numbers with two principle components (2D)
def plt_pca_MNIST(pca_df):
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
def plt_pca__MNIST_3d(pca_df):
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


def load_orl(path):
    print("Load ORL data")
    orl_data = sio.loadmat(path + "orl_data.mat")
    orl_lbls = sio.loadmat(path + "orl_lbls.mat")

    return orl_data, orl_lbls


# Convert ORL images to a 1x400 vector with each element containing a 40x30 array
# This is a stupid way to do it, but my brain didn't work at the time
def convert_orl_to_vector(orl_data):
    a = orl_data['data']

    height = 39
    column = 0
    pic_index = 0
    j = 0
    image = np.zeros([40, 30])
    images = list(range(1, 401))

    for k in range(len(images)):
        for i in range(len(a)):
            image[j][column] = a[i][pic_index]
            j = j + 1
            if i >= height:
                column = column + 1
                height = height + 40
                j = 0
        images[k] = image
        height = 39
        column = 0
        pic_index = pic_index + 1
        image = np.zeros([40, 30])

        orl = np.array(images)

    return orl

# TODO make this have the same structure as nnc_classify
# Perform Nearest class centroid classifier of original data (NOT PCA)
def ncc_classify(X_tr, y_tr, X_te, dataset):
    if (dataset == 'orl'):
        nsamples, nx, ny = X_tr.shape
        d2_images = X_tr.reshape((nsamples, nx * ny))

        X_train, X_test, y_train, y_test = train_test_split(
            d2_images, y_tr['lbls'], test_size=0.33, random_state=42)

    model = NearestCentroid()
    model.fit(X_train, y_train)

    if (dataset == 'orl'):
        y_pred = model.predict(X_test)

    y_pred = model.predict(X_te)

    return model, y_pred


# Perform Nearest neighbor classifier of original data (NOT PCA)
def nnc_classify(dataset):

    if dataset != "MNIST" and dataset != "ORL":
        print("Typo in nnc_classify")
        return -1, -1

    if dataset == "ORL":
        path = "C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/"
        orl_data, orl_lbls = load_orl(path)
        orl_data = convert_orl_to_vector(orl_data)

        model, y_pred = nnc_orl(orl_data, orl_lbls)

    if dataset == "MNIST":
        path = "C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/"
        X_train, y_train, X_test, y_test = load_idx(path)

        model = neighbors.KNeighborsClassifier(n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        make_confusion_matrix(model, X_train, y_train, X_test, y_test)

    return model, y_pred


def nnc_orl(data, labls):

    # Reshape data
    nsamples, nx, ny = data.shape
    d2_images = data.reshape((nsamples, nx * ny))

    # Split data into 70% training 30% testing. The stratify option makes sure to split each class the same way, i.e
    # use 7 pictures of class 1 for training and 3 pictures for testing.
    X_train, X_test, y_train, y_test = train_test_split(
        d2_images, labls['lbls'], test_size=0.30, random_state=41, stratify=labls['lbls'])

    # Make sure labels are in correct format using LabelEncoder
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)

    # Train model
    model = neighbors.KNeighborsClassifier(n_jobs=-1)

    new_params = find_best_parameters(model, X_train, y_train)


    model = model.set_params(n_neighbors = new_params['n_neighbors'], algorithm = new_params['algorithm'])

    model.fit(X_train, y_train)

    # Test model
    y_pred = model.predict(X_test)

    # How did we do?
    y_test = encoder.fit_transform(y_test)
    make_confusion_matrix(model, X_train, y_train, X_test, y_test)

    # Return stuff
    return model, y_pred

# TODO Fejl med gridsearch. score bruges ikke rigtigt.
def find_best_parameters(model, X_train, y_train):
    params = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'algorithm': ['ball_tree', 'auto', 'kd_tree', 'brute']
    }
    score = ['precision', 'recall']

    clf = GridSearchCV(model, param_grid=params, cv=5, scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    mean = clf.cv_results_['mean_test_score']
    print(mean)

    print(clf.best_params_)

    return clf.best_params_


if __name__ == '__main__':

    # a = input("Type in either 'MNIST' or 'ORL'.")
    model, y_pred = nnc_classify('ORL')



    print(model)

    print(y_pred)
