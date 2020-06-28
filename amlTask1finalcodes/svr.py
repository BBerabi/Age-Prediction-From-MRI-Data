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
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.svm import SVR
import auxilary

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
X_train, y_train = auxilary.OutlierDetectionIsolationForest(X_train, y_train, percentageOutlier = 0.02)
print("Shape after outlier detection: ", X_train.shape)

inputDim = 185

#Feature Selection
featureSelection = SelectKBest(f_regression, k = inputDim)
X_train = featureSelection.fit_transform(X_train, y_train)
scores = featureSelection.scores_


#Stardardize the training nad validation data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train = np.nan_to_num(X_train, nan = 0)


k = 10

kernel = 'rbf'
gammas = 'scale'
cvalues = 140
cvalues = np.arange(125, 140,25)
#epsilons = np.arange(0.05, 0.1, 0.01)

bestC = 0
bestEps = 0
bestScore = np.NINF

print("Computation starts")
print(k)
for cvalue in cvalues:
    print("Cvalue: ", cvalue)
    #for cvalue in cvalues:
        #print("C: ", cvalue)
    kf = KFold(n_splits = k, shuffle = False)
    scores = []
    for train_index, test_index in kf.split(X_train,y_train):

        X_traincv, Y_train = X_train[train_index],y_train[train_index]
        X_test, Y_test = X_train[test_index], y_train[test_index]

        svr = SVR(kernel = kernel, C = cvalue, gamma = 'scale', coef0 = 0, epsilon=0.1)
        svr.fit(X_traincv, Y_train)
        y_pred = svr.predict(X_test)
        score = svr.score(X_test, Y_test)
        scores.append(score)

        
    averagedScore = (sum(scores)/k)
    if averagedScore > bestScore:
        bestC = cvalue
        #bestEps = epsilon
        bestScore = averagedScore


print("best score is ", bestScore)
#print("best degree is ", bestDegree)
print("best cvalue is ", bestC)
print('best eps: ', bestEps)
#print("best gamma is ", bestGamma)



svr = SVR(kernel = 'rbf', C = bestC, gamma = 'scale', coef0 = 0, epsilon=0.1)
svr.fit(X_train, y_train)

X_test = pd.read_csv(r"C:\Users\berka\Desktop\ETH_ZÜRICH_MASTER\Third Semester\AML\Projects\task1\X_test.csv")
del X_test['id']
X_test = scaler.fit_transform(X_test)
X_test = np.nan_to_num(X_test, nan = 0)
X_test = featureSelection.transform(X_test)

y_pred_test = svr.predict(X_test)
auxilary.createSubmissionFiles(y_pred_test)


