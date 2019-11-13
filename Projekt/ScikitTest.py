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
from sklearn.metrics import classification_report, SCORERS, make_scorer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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


def load_orl(path):
    print("Load ORL data")
    orl_data = sio.loadmat(path + "orl_data.mat")
    orl_lbls = sio.loadmat(path + "orl_lbls.mat")

    X = np.array(orl_data['data'])
    y = np.array(orl_lbls['lbls'])
    y = np.transpose(y)

    y = y.reshape(400)

    X = np.transpose(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=41, stratify=y)

    y_train = np.transpose(y_train)
    y_test = np.transpose(y_test)

    return X_train, y_train, X_test, y_test

# TODO Something is wrong here... "Could not decode [1 2 3... 40] to [1 2 3... 40] labels."
def make_confusion_matrix(model, X_train, y_train, X_test, y_test):

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)

    classes = list()

    for a in np.unique(y_train):
        classes.append(a)

    #For some reason it gives an error if not done this way...
    if len(classes) > 10:
        classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
               24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    else:
        classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)

    cm = ConfusionMatrix(model, classes=classes)
    cm.fit(X_train, y_train)

    encoder.fit(y_train)
    y_test = encoder.transform(y_test)

    cm.score(X_test, y_test)

    cm.show()

    return


def scale_mnist(Xtrain, Xtest):
    Xtrain = StandardScaler().fit_transform(Xtrain)
    Xtest = StandardScaler().fit_transform(Xtest)
    return Xtrain, Xtest


# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# Perform PCA to X and concat with labels
def perform_pca(X, y, dim):
    pca = PCA(n_components=dim)

    principalComponents = pca.fit_transform(X)

    if dim == 2:
        principalDf = pd.DataFrame(data=principalComponents, columns=['Principle component 1', 'Principle component 2'])

    if dim == 3:
        principalDf = pd.DataFrame(data=principalComponents, columns=['Principle component 1', 'Principle component 2',
                                                                      'Principle component 3'])

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
# Perform Nearest class centroid classifier of original data
def ncc_classify(X_train, y_train, X_test, y_test):

    model = NearestCentroid()
    params = {
        'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'algorithm': ['kd_tree', 'auto', 'ball_tree', 'brute']
    }

    new_params = find_best_parameters(model, X_train, y_train, params)
    model.set_params(n_neighbors=new_params['n_neighbors'], algorithm=new_params['algorithm'])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    make_confusion_matrix(model, X_train, y_train, X_test, y_test)

    return model, y_pred


# Perform Nearest neighbor classifier of original data (NOT PCA)
def nnc_classify(X_train, y_train, X_test, y_test):

        params = {
            'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'algorithm': ['kd_tree', 'auto', 'ball_tree', 'brute']
        }

        model = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='auto')
        new_params = find_best_parameters(model, X_train[:100], y_train[:100], params)

        model = model.set_params(n_neighbors=new_params['n_neighbors'], algorithm=new_params['algorithm'])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        make_confusion_matrix(model, X_train, y_train, X_test, y_test)

        return model, y_pred


def ncc_sub(k, X_train, y_train, X_test):
    # Make dict with classes and k number empty lists
    centroids = {int(a) : list() for a in np.unique(y_train)}
    for i in range(len(X_train)):
        centroids[y_train[i]].append(X_train[i])

    # Make K centroids for each class
    for key in centroids:
        centroids[key] = KMeans(n_clusters=k, init='random', n_init=10).fit(np.array(centroids[key])).cluster_centers_

    labels = list()

    # Calculate distance between subclass and datapoint. Add lowest distance subclass to labels list.
    for i in range(len(X_test)):
        minDist = np.inf
        popflag = False
        for key in centroids:
            for j in range(centroids[key].shape[0]):
                d = np.sqrt(np.linalg.norm(X_test[i] - centroids[key][j]))
                if d < minDist:
                    minDist = d
                    if popflag:
                        labels.pop(i)
                        popflag = False
                    # This is done to see which subclass is picked. Can only be seen when debugging
                    labels.insert(i, str(key) + "_" + str(j))
                    popflag = True

        # Remove subclasses and make int
        labels[i] = int(labels[i].split('_')[0])

    return labels


def perceptron_bp(X_train, y_train, eta):

    # Initial weights and bias
    w = np.zeros((len(np.unique(y_train)), X_train.shape[1]))
    x = np.ones((len(X_train), 1))
    w0 = np.zeros((len(w), 1))
    li = np.zeros((40, len(X_train)))

    # Make decision function more compact
    x_tilde = np.concatenate((x, X_train),1)
    w_tilde = np.concatenate((w0, w), 1)

    w_tilde = np.random.rand(w_tilde.shape[0], w_tilde.shape[1]) - 0.5

    g = w_tilde.dot(np.transpose(x_tilde))

    # Determine classes
    for i in range(li.shape[1]):
        for j in range(40):
            if y_train[i] == j:
                li[j][i] = 1
            else:
                li[j][i] = -1

    f = np.transpose(li).dot(g)
    return


# TODO Fejl med gridsearch. score bruges ikke rigtigt.
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
def find_best_parameters(model, X, y, params):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0, stratify=y)

    scores = ["recall"]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(model, params, cv=5)
        clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    return clf.best_params_


if __name__ == '__main__':
    '''
    path = "C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/"
    X_train,  y_train, X_test, y_test = load_idx(path)
    '''
    path = "C:/Users/stinu/Desktop/RandomSkole/ODA/Projekt/samples/"
    X_train, y_train, X_test, y_test = load_orl(path)

    perceptron_bp(X_train, y_train, 1)


    # cm = ncc_sub(2,X_train, y_train, X_test)
    # model, y_pred = nnc_classify(X_train, y_train, X_test)
    # ncc_classify(X_train,y_train,X_test, y_test)
    # pca = PCA(n_components=2)
    # pca_X_train = pca.fit_transform(X_train)
    # pca_X_test = pca.fit_transform(X_test)

    # ncc_classify(X_train, y_train, X_test, y_test)
    # cm = ncc_sub(2, X_train, y_train, X_test)

    print(1)
