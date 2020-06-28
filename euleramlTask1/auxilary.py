import numpy as np
import numpy.linalg as LA
from scipy import linalg as LA2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
seedValue = 1
np.random.seed(seedValue)
import random
random.seed(seedValue)
from sklearn.ensemble import IsolationForest

from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

import tensorflow as tf
import keras
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import BatchNormalization	
from keras.utils import np_utils
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers

import auxilary
import logging
logging.getLogger('tensorflow').disabled = True

DEBUG = False

def getModel(dimensionOfInput):
    model = Sequential()
        
    model.add(Dense(128, input_dim = dimensionOfInput))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.30))

    model.add(Dense(256, use_bias=True,))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.30))

    model.add(Dense(256, use_bias=True,))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.30))

    model.add(Dense(256, use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.30))

    model.add(Dense(256, use_bias=True))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.30))


    model.add(Dense(1, activation = 'linear'))
    return model

def deepModel(dimensionOfInput):

    model = Sequential()
        
    model.add(Dense(256, input_dim = dimensionOfInput))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(LeakyReLU(alpha = 0.1))
    model.add(Dropout(0.35))

    model.add(Dense(128, use_bias=False,))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(LeakyReLU(alpha = 0.1))
    model.add(Dropout(0.35))

    # model.add(Dense(64, use_bias=False,))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))

    # model.add(Dense(32, use_bias=False))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.35))

    # model.add(Dense(16, use_bias=False))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.35))



    model.add(Dense(1, activation = 'linear'))
    return model


def zeroMean(data):
    colMean = np.nanmean(data, axis = 0)
    if DEBUG:
        print("The mean of features: ", colMean)
    data -= colMean
    data = np.nan_to_num(data, nan = 0)
    return data



def OutlierDetectionEuclideanMetric(data, ydata):
    neigh = NearestNeighbors(n_neighbors=15)
    neigh.fit(data)

    distancesToNN, indices = neigh.kneighbors(data, return_distance=True)
    meanDistanceOfNN = np.mean(distancesToNN, axis = 1)

    factor = 1.05
    mean_dist_tot = np.mean(meanDistanceOfNN)
    threshold = factor * mean_dist_tot
    index_to_be_removed = meanDistanceOfNN > threshold

    if DEBUG:
        print("total mean distance: ", mean_dist_tot)
        print("threshold:", threshold)
        print("Number of indices to be removed:", index_to_be_removed.sum())

    data = np.delete(data, np.where(index_to_be_removed), axis=0)
    ydata = np.delete(ydata, np.where(index_to_be_removed), axis=0)
    #print("New number of data after droping points with too long mean_dist from KNN:", data.shape[0])
    #print("New number of y after droping points with too long mean_dist from KNN:", ydata.shape[0])

    return data, ydata


def OutlierDetectionIsolationForest(data, labels, percentageOutlier):
    
    clf = IsolationForest( behaviour = 'new', max_samples=0.99, random_state = 1, contamination= percentageOutlier)
    preds = clf.fit_predict(data)

    indicesToRemove = np.argwhere(preds == -1)
    numberOfOutliers = np.count_nonzero(preds == -1)
    print("Number Of Outliers:", numberOfOutliers)
    data = np.delete(data, indicesToRemove, axis = 0)
    labels = np.delete(labels, indicesToRemove)

    return data, labels



def createSubmissionFiles(y_predictions):
    output = pd.DataFrame()
    output.insert(0, 'y', y_predictions)
    A = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task1\X_test.csv")
    output.index = A.index
    output.index.names = ['id']
    output.to_csv("output")

    outputRounded = pd.DataFrame()
    outputRounded.insert(0, 'y', np.rint(y_predictions))
    outputRounded.index = A.index
    outputRounded.index.names = ['id']
    outputRounded.to_csv("outputRounded")
    print("Submission files are succesfully created")

def coefficientofdetermination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# # Outlier detection
#X_train, y_train = auxilary.OutlierDetectionEuclideanMetric(X_train, y_train)

# #%%
#Apply PCA

#pca = PCA(n_components=0.7, whiten = True, svd_solver='full')
#pca = PCA(n_components = 400) ######deci this value according to eigenvalue spectrum
#X_train = pca.fit_transform(X_train)
#X_train = auxilary.zeroMean(X_train)



# CovMatrix = np.cov(X_train, rowvar=False)
# print("Shape of cov: ", CovMatrix.shape)
# D, U = LA2.eigh(CovMatrix)
# print("Shape of U: ", U.shape)
# print("Shape of D:", D.shape)
# D = np.real(D)
# D = np.sort(D)
#print(D)


# # and look at NaNs
# if DEBUG:
#     print("\nTotal Number of NaN in X_train:", np.isnan(X_train).sum() )
