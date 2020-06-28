import numpy as np
import pandas as pd
import tensorflow as tf

import numpy.linalg as LA
import keras
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


trainDataX = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task1\X_train.csv")
trainDataY = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task1\y_train.csv")
testDataX = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task1\X_test.csv")

#print(trainDataX.values.shape)
#print(trainDataX)

def zeroMean(data):
    colMean = np.nanmean(data, axis = 0)
    data -= colMean
    data = np.nan_to_num(data)
    return data

#k is the number of dimensions to keep
def dimensionReductionPCA(data, k):
    pca = PCA(k)

    return pca.fit_transform(data)

def Scatter2D(dataX, color):
    if dataX.shape[1] != 2:
        print("The dimension of samples has to be 2, not plotting")
        return
    plt.scatter(dataX[:,0], dataX[:,1],  c = color.reshape(color.shape[0],) )
    plt.colorbar()
    plt.show()


scaler = StandardScaler()

del trainDataX['id']


X_train = trainDataX.values
X_train = zeroMean(X_train)
X_train = scaler.fit_transform(X_train)

print(np.mean(X_train, axis = 0))

#print(np.mean(X_train).sum())