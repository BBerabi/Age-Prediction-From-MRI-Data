
#%%
import numpy as np
import numpy.linalg as LA
from scipy import linalg as LA2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
seedValue = 42
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
from sklearn.feature_selection import SelectKBest, f_regression

import tensorflow as tf
tf.compat.v1.set_random_seed(seedValue)
tf.random.set_random_seed(seedValue)
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

DEBUG = True
TRAIN = True

X_train = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task1\X_train.csv")
y_train = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task1\y_train.csv")
X_test = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task1\X_test.csv")
X_train = X_train.drop(columns='id', axis=1)
y_train = y_train.drop(columns='id', axis=1)
y_train = y_train.values

#Fill in nan values with mean of that particular feature
X_train = np.nan_to_num(X_train, nan = np.nanmean(X_train, axis = 0))

#Remove Outliers
X_train, y_train = auxilary.OutlierDetectionIsolationForest(X_train, y_train, percentageOutlier = 'auto')
print("Shape after outlier detection: ", X_train.shape)

inputDim = 150

#Feature Selection
featureSelection = SelectKBest(f_regression, k = inputDim)
X_train = featureSelection.fit_transform(X_train, y_train)
scores = featureSelection.scores_
print("Shape after feature selection: ", X_train.shape)
#print("Scores of features:\n",scores)
#print("Number of scores: ", scores.shape)

scores = np.sort(scores)[::-1]
axis = np.arange(1,833)
scores = np.nan_to_num(scores, nan = 0)
total = np.sum(scores)
print("total: ", total)
# plt.plot(axis, scores, 'ro',marker = '+', markersize = 0.1)
# plt.plot(axis, scores/total, 'ro')
# plt.show()

#Split the data in to training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=seedValue)

#Stardardize the training nad validation data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train = np.nan_to_num(X_train, nan = 0)
X_val = scaler.fit_transform(X_val)
X_val = np.nan_to_num(X_val, nan = 0)


#model = auxilary.getModel(inputDim)
model = auxilary.getModel(inputDim)
es = EarlyStopping(monitor='val_coefficientofdetermination',mode='max',verbose=1,patience=700)
mc = ModelCheckpoint('best_model.h5',monitor='val_coefficientofdetermination',mode='max',verbose=1,save_best_only=True)
opt = keras.optimizers.Adam(lr = 0.001)

if TRAIN:
        
    model.compile(loss = 'mse', optimizer = opt, metrics = [auxilary.coefficientofdetermination])
    model.fit(X_train, y_train,  validation_data = (X_val, y_val), epochs = 2000, batch_size = 32, callbacks = [es, mc])

    bestModel = load_model('best_model.h5', custom_objects = {'coefficientofdetermination' : auxilary.coefficientofdetermination})

    del X_test['id']
    X_test = scaler.fit_transform(X_test)
    X_test = np.nan_to_num(X_test, nan = 0)
    X_test = featureSelection.transform(X_test)

    y_predictions = bestModel.predict(X_test)
    y_predictions = np.reshape(y_predictions, y_predictions.shape[0])

    auxilary.createSubmissionFiles(y_predictions)


model.predict(X_test)