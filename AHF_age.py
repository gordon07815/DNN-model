#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 13:57:26 2021

@author: gordon07815
"""
import pandas as pd
import numpy as np
import scipy.stats as st
import math
import re
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import ADASYN, SMOTE

ahf_data = pd.read_excel("V2 lab.xlsx")
#ccl_data = pd.read_excel("CCI_20211005.xls")
#ma_data = pd.read_excel("MA_20211005.xls")
Normal_age_sex = pd.read_excel("Normal_age_sex.xlsx")
Normal_age_sex.N_Age = np.array(Normal_age_sex.N_Age.astype('int64')).reshape(1545,1)
Normal_age_sex.M_sex = Normal_age_sex.M_sex.str.replace(r"[男M]","1",regex = True)
Normal_age_sex.M_sex = np.array(Normal_age_sex.M_sex.str.replace(r"[女F]","2",regex = True)).reshape(1545,1)
age_sex = pd.concat((Normal_age_sex.N_Age, Normal_age_sex.M_sex), axis=1)
age_sex = np.concatenate((age_sex.to_numpy(), np.array([0]*1545).reshape(1545,1)), axis=1)

age_ahf = []
for i in range(433):
    if(str(ahf_data.檢查日[i]).find("NaT") >= 0):
        if(str(ahf_data.出生日[i]).find("NaT") >= 0):
            age_ahf = np.append(age_ahf, "0")
        else:
            age_ahf = np.append(age_ahf, int(2007) - int(str(ahf_data.出生日[i])[:4]))
    elif(str(ahf_data.出生日[i]).find("NaT") >= 0):
        age_ahf = np.append(age_ahf, "0")
    else:
        age_ahf = np.append(age_ahf, int(str(ahf_data.檢查日[i])[:4]) - int(str(ahf_data.出生日[i])[:4]))

age_ahf = age_ahf.reshape(433,1)
sex_ahf = ahf_data.sex.str.replace('女','2')
sex_ahf = sex_ahf.str.replace('男','1')
sex_ahf = np.array(sex_ahf.str.replace(r'\D+','0',regex=True)).reshape(433,1)
for i in range(len(sex_ahf)):
    if(math.isnan(sex_ahf[i:][0])):
        print(i)
        sex_ahf[i][0] = 0

age_ahf = np.concatenate((age_ahf, sex_ahf), axis=1)
age_ahf = np.concatenate((age_ahf, np.array([1]*433).reshape(433,1)), axis=1)
age_sex = np.concatenate((age_sex, age_ahf))

import scipy.stats as st
a,b,c,d = [0]*6, [0]*6, [0]*6, [0]*6
odd, p = [0]*6, [0]*6
a[0]=sum(age_sex[np.where(age_sex[:1545,0].astype('float64')<50),1][0].astype('int')==1)
b[0]=sum(age_sex[np.where(age_sex[:1545,0].astype('float64')<50),1][0].astype('int')==2)
c[0]=sum(age_sex[np.where(age_sex[1545:,0].astype('float64')<50)[0]+1545,1].astype('int')==1)
d[0]=sum(age_sex[np.where(age_sex[1545:,0].astype('float64')<50)[0]+1545,1].astype('int')==2)
odd[0], p[0] = st.fisher_exact(np.array([[a[0],b[0]],[c[0],d[0]]]))
for i in range(1,6):
    a[i]=sum(age_sex[np.where(age_sex[:1545,0].astype('float64')<50+i*10),1][0].astype('int')==1)-sum(age_sex[np.where(age_sex[:1545,0].astype('float64')<40+i*10),1][0].astype('int')==1)
    b[i]=sum(age_sex[np.where(age_sex[:1545,0].astype('float64')<50+i*10),1][0].astype('int')==2)-sum(age_sex[np.where(age_sex[:1545,0].astype('float64')<40+i*10),1][0].astype('int')==2)
    c[i]=sum(age_sex[np.where(age_sex[1545:,0].astype('float64')<50+i*10)[0]+1545,1].astype('int')==1)-sum(age_sex[np.where(age_sex[1545:,0].astype('float64')<40+i*10)[0]+1545,1].astype('int')==1)
    d[i]=sum(age_sex[np.where(age_sex[1545:,0].astype('float64')<50+i*10)[0]+1545,1].astype('int')==2)-sum(age_sex[np.where(age_sex[1545:,0].astype('float64')<40+i*10)[0]+1545,1].astype('int')==2)
    odd[i], p[i] = st.fisher_exact(np.array([[a[i],b[i]],[c[i],d[i]]]))
    


X, y = NearMiss().fit_resample(age_sex[:,:2].astype('float64'), age_sex[:,2].astype('int'))

X, y = SMOTE().fit_resample(X=age_sex[:,:2].astype('float64'), y=age_sex[:,2].astype('int'))

X, y = ADASYN().fit_resample(age_sex[:,:2].astype('float64'), age_sex[:,2].astype('int'))

a,b,c,d = [0]*6, [0]*6, [0]*6, [0]*6
odd, p = [0]*6, [0]*6
a[0]=sum(X[np.where(X[:433,0].astype('float64')<50),1][0].astype('int')==1)
b[0]=sum(X[np.where(X[:433,0].astype('float64')<50),1][0].astype('int')==2)
c[0]=sum(X[np.where(X[433:,0].astype('float64')<50)[0]+433,1].astype('int')==1)
d[0]=sum(X[np.where(X[433:,0].astype('float64')<50)[0]+433,1].astype('int')==2)
odd[0], p[0] = st.fisher_exact(np.array([[a[0],b[0]],[c[0],d[0]]]))
for i in range(1,6):
    a[i]=sum(X[np.where(X[:433,0].astype('float64')<50+i*10),1][0]==1)-sum(X[np.where(X[:433,0].astype('float64')<40+i*10),1][0]==1)
    b[i]=sum(X[np.where(X[:433,0].astype('float64')<50+i*10),1][0]==2)-sum(X[np.where(X[:433,0].astype('float64')<40+i*10),1][0]==2)
    c[i]=sum(X[np.where(X[433:,0].astype('float64')<50+i*10)[0]+433,1]==1)-sum(X[np.where(X[433:,0].astype('float64')<40+i*10)[0]+433,1]==1)
    d[i]=sum(X[np.where(X[433:,0].astype('float64')<50+i*10)[0]+433,1]==2)-sum(X[np.where(X[433:,0].astype('float64')<40+i*10)[0]+433,1]==2)
    odd[i], p[i] = st.fisher_exact(np.array([[a[i],b[i]],[c[i],d[i]]]))

a,b,c,d = [0]*6, [0]*6, [0]*6, [0]*6
odd, p = [0]*6, [0]*6
a[0]=sum(X[np.where(X[:1545,0].astype('float64')<50),1][0]==1)
b[0]=sum(X[np.where(X[:1545,0].astype('float64')<50),1][0]==2)
c[0]=sum(X[np.where(X[1545:,0].astype('float64')<50)[0]+1545,1]==1)
d[0]=sum(X[np.where(X[1545:,0].astype('float64')<50)[0]+1545,1]==2)
odd[0], p[0] = st.fisher_exact(np.array([[a[0],b[0]],[c[0],d[0]]]))
for i in range(1,6):
    a[i]=sum(X[np.where(X[:1545,0].astype('float64')<50+i*10),1][0]==1)-sum(X[np.where(X[:1545,0].astype('float64')<40+i*10),1][0]==1)
    b[i]=sum(X[np.where(X[:1545,0].astype('float64')<50+i*10),1][0]==2)-sum(X[np.where(X[:1545,0].astype('float64')<40+i*10),1][0]==2)
    c[i]=sum(X[np.where(X[1545:,0].astype('float64')<50+i*10)[0]+1545,1]==1)-sum(X[np.where(X[1545:,0].astype('float64')<40+i*10)[0]+1545,1]==1)
    d[i]=sum(X[np.where(X[1545:,0].astype('float64')<50+i*10)[0]+1545,1]==2)-sum(X[np.where(X[1545:,0].astype('float64')<40+i*10)[0]+1545,1]==2)
    odd[i], p[i] = st.fisher_exact(np.array([[a[i],b[i]],[c[i],d[i]]]))


AHF_data = pd.read_csv('AHF_noted.csv', header=None)
AHF_data = pd.concat([AHF_data[range(1000)], AHF_data[1002]],axis=1)
AHF_data= AHF_data.rename(columns={1002:1000})
array = np.empty([1,1001])
for i in range(len(AHF_data)):
    if(re.search(r'AHF\d*',AHF_data[0][i])):
        a = re.search(r'AHF\d*',AHF_data[0][i])
        if(ahf_data['編號'].str.contains(a.group(0)).any()):
            b = np.append(AHF_data.iloc[i,:998], age_ahf[pd.Index(ahf_data['編號']).get_loc(a.group(0))])[:1001].reshape(1,1001)
            array = np.append(array,b,axis=0)
            
normal_data = pd.read_csv('normal_noted.csv', header=None)   
for i in range(len(normal_data)):
    if(Normal_age_sex.ID.astype(str).str.contains(normal_data.iloc[i,0].astype(str)).any()):
        j = np.where(Normal_age_sex.ID==normal_data.iloc[i,0])[0][0]
        if(i==0):
            normal_data_A_S = np.append(normal_data.iloc[i,:998], 
            np.append(Normal_age_sex.N_Age[j],
            np.append(Normal_age_sex.M_sex[j], normal_data.iloc[i,1000]))).reshape(1,1001)
        else:    
            normal_data_A_S = np.concatenate((
            normal_data_A_S,
            np.append(normal_data.iloc[i,:998], 
            np.append(Normal_age_sex.N_Age[j],
            np.append(Normal_age_sex.M_sex[j], normal_data.iloc[i,1000]))).reshape(1,1001)), axis=0)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,KFold, RepeatedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from numpy import mean, std, arange
import matplotlib.pyplot as plt
array = np.append(array[:,1:1001], normal_data_A_S[:,1:1001], axis=0)
data = np.append(AHF_data[range(1,1001)], normal_data[range(1,1001)], axis=0)
X_train, X_test, y_train, y_test = train_test_split(age_sex[:,:2], age_sex[:,2], test_size = 0.2, random_state = 1)
X_train, X_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size = 0.2, random_state = 1)
X_train, X_test, y_train, y_test = train_test_split(array[:,1:-1], array[:,-1], test_size = 0.2, random_state = 1)
#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
y_train = y_train.reshape(len(y_train),).astype('int')
y_test = y_test.reshape(len(y_test),).astype('int')

clf = LogisticRegression(random_state = 0, solver='liblinear')
grid = {"penalty":["l1","l2"], "C":arange(0.1,1.1,0.1)}
cv = KFold(n_splits=10, random_state=0, shuffle=True)
hyper = GridSearchCV(clf, grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
result = hyper.fit(X_train, y_train)
print("Accuracy: %.3f" % result.best_score_) 
print("Parameters: ", result.best_params_)

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):

    scores_mean = result.cv_results_['mean_test_score']
    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    _, ax = plt.subplots(1,1)

    a = [2 * n for n in range(10)]
    b = [(2 * n + 1) for n in range(10)]
    ax.plot(grid_param_1, scores_mean[a], '-o', label= name_param_2 + ': l1')
    ax.plot(grid_param_1, scores_mean[b], '-o', label= name_param_2 + ': l2')
    ax.set_title("Grid of logistic regression", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.savefig('Grid_logisticregression.png')

plot_grid_search(result.cv_results_, result.param_grid['C'], result.param_grid['penalty'], 'C', 'Penalty')

scores = np.empty([0,2])
clf = LogisticRegression(random_state = 0, C=0.6, penalty='l2', solver='liblinear')
for i in range(10):
    cv = KFold(n_splits=10, random_state=i, shuffle=True)
    score = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=cv)
    scores = np.append(scores, [[mean(score), std(score)]], axis=0)
    print("Accuracy: %.3f(%.3f)" % (scores[i,0], scores[i,1]))
plt.subplots(figsize=(8,6))
plt.errorbar(range(1,11), scores[:,0], scores[:,1], fmt='-o', color='b', ecolor='r', elinewidth=2, capsize=4)
plt.ylim([0.7,.9])
plt.savefig('10-folds.png')

from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, roc_auc_score, auc
def plot_ROC_PRC(clf, name):
    y_score = clf.fit(X_train, y_train).decision_function(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    
    plt.figure()
    plt.plot(fpr,tpr,color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig(name+'_ROC.png')
    
    prob = clf.predict_proba(X_test)
    prob = prob[:,1]
    precision, recall, _ = precision_recall_curve(y_test, prob)
    f1, au= f1_score(y_test, clf.predict(X_test)), auc(recall, precision)
    print('Logistic: f1=%.3f auc=%.3f' % (f1, au))
    no_skill = len(y_test[y_test==1]) / len(y_test)
    
    plt.figure()
    plt.plot([0,1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.scatter(recall, precision, color=['darkorange'], lw=2, marker='.', label='Logistic (area = %.2f)' % au)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(name+'_PR curve.png')

plot_ROC_PRC(clf,'Age&Sex')
plot_ROC_PRC(clf,'PWV')
plot_ROC_PRC(clf,'PWV_Age&Sex')


#clf = LogisticRegression(random_state = 0, C=0.9, penalty='l1', solver='liblinear')
def cm(clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    c = confusion_matrix(y_test, y_pred)
    print(c)
    print(accuracy_score(y_test, y_pred))
cm(clf)




X = AHF_data.iloc[:, 0:-1].values
y = AHF_data.iloc[:, -1].values
X = np.append(X, normal_data.iloc[:,0:-1].values, axis=0)
y = np.append(y, normal_data.iloc[:,-1].values, axis=0)

from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Flatten, LSTM, BatchNormalization, Dropout, GRU, Concatenate, Activation
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from random import randint
#from itertools import cycle

#record = []

tpr = dict()
fpr = dict()
roc_auc = dict()
n = 866
#y_mod = np.asarray([0]*500+[1]*433).astype('int')
for i in range(10):
    #X_mod, _ = train_test_split(age_sex[:1545,:2], test_size = 0.7, random_state = i)
    #X_mod = np.append(X_mod, age_sex[1545:,:2], axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = i+1)
    ''' X_train = np.append(X[0:int(i*n/10),:],X[int((i+1)*n/10):,:],axis=0)
    X_test = X[int(i*n/10):int((i+1)*n/10),:]
    y_train = np.append(y[0:int(i*n/10)],y[int((i+1)*n/10):],axis=0)
    y_test = y[int(i*n/10):int((i+1)*n/10)]'''
    X_train = X_train.reshape(len(X_train),2,1).astype('float64')
    X_test = X_test.reshape(len(X_test),2,1).astype('float64')
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    inputs = Input(shape=(2,1))
    '''
    x1 = Conv1D(256, 2, activation=('relu'))(inputs)
    x2 = Conv1D(256, 3, activation=('relu'))(inputs)
    x3 = Conv1D(128, 2, activation=('relu'))(inputs)
    x4 = Conv1D(128, 3, activation=('relu'))(inputs)
    x5 = Conv1D(256, 4, activation=('relu'))(inputs)
    x6 = Conv1D(128, 4, activation=('relu'))(inputs)
    x7 = Conv1D(64, 2, activation=('relu'))(inputs)
    x8 = Conv1D(64, 3, activation=('relu'))(inputs)
    x9 = Conv1D(64, 4, activation=('relu'))(inputs)
    x10 = Conv1D(32, 2, activation=('relu'))(inputs)
    x11 = Conv1D(32, 3, activation=('relu'))(inputs)
    x12 = Conv1D(32, 4, activation=('relu'))(inputs)
    x1 = MaxPool1D()(x1)
    x2 = MaxPool1D()(x2)
    x3 = MaxPool1D()(x3)
    x4 = MaxPool1D()(x4)
    x5 = MaxPool1D()(x5)
    x6 = MaxPool1D()(x6)
    x7 = MaxPool1D()(x7)
    x8 = MaxPool1D()(x8)
    x9 = MaxPool1D()(x9)
    x10 = MaxPool1D()(x10)
    x11 = MaxPool1D()(x11)
    x12 = MaxPool1D()(x12)
    x1 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x1)
    x2 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x2)
    x3 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x3)
    x4 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x4)
    x5 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x5)
    x6 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x6)
    x7 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x7)
    x8 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x8)
    x9 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x9)
    x10 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x10)
    x11 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x11)
    x12 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x12)
    x1 = Conv1D(256, 2, activation=('relu'))(x1)
    x2 = Conv1D(256, 3, activation=('relu'))(x2)
    x3 = Conv1D(128, 2, activation=('relu'))(x3)
    x4 = Conv1D(128, 3, activation=('relu'))(x4)
    x5 = Conv1D(256, 4, activation=('relu'))(x5)
    x6 = Conv1D(128, 4, activation=('relu'))(x6)
    x7 = Conv1D(64, 2, activation=('relu'))(x7)
    x8 = Conv1D(64, 3, activation=('relu'))(x8)
    x9 = Conv1D(64, 4, activation=('relu'))(x9)
    x10 = Conv1D(32, 2, activation=('relu'))(x10)
    x11 = Conv1D(32, 3, activation=('relu'))(x11)
    x12 = Conv1D(32, 4, activation=('relu'))(x12)
    x1 = MaxPool1D()(x1)
    x2 = MaxPool1D()(x2)
    x3 = MaxPool1D()(x3)
    x4 = MaxPool1D()(x4)
    x5 = MaxPool1D()(x5)
    x6 = MaxPool1D()(x6)
    x7 = MaxPool1D()(x7)
    x8 = MaxPool1D()(x8)
    x9 = MaxPool1D()(x9)
    x10 = MaxPool1D()(x10)
    x11 = MaxPool1D()(x11)
    x12 = MaxPool1D()(x12)
    x1 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x1)
    x2 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x2)
    x3 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x3)
    x4 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x4)
    x5 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x5)
    x6 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x6)
    x7 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x7)
    x8 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x8)
    x9 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x9)
    x10 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x10)
    x11 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x11)
    x12 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x12)
    x1 = Dense(units=32, activation=('relu'))(x1)
    x2 = Dense(units=32, activation=('relu'))(x2)
    x3 = Dense(units=32, activation=('relu'))(x3)
    x4 = Dense(units=32, activation=('relu'))(x4)
    x5 = Dense(units=32, activation=('relu'))(x5)
    x6 = Dense(units=32, activation=('relu'))(x6)
    x7 = Dense(units=32, activation=('relu'))(x7)
    x8 = Dense(units=32, activation=('relu'))(x8)
    x9 = Dense(units=32, activation=('relu'))(x9)
    x10 = Dense(units=32, activation=('relu'))(x10)
    x11 = Dense(units=32, activation=('relu'))(x11)
    x12 = Dense(units=32, activation=('relu'))(x12)
    x = Concatenate(axis=1)([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12])
    #x = Dense(units=512, activation=('relu'))(inputs)
    '''
    x = Dense(units=64, activation=('relu'))(inputs)
    x = Dense(units=8, activation=('relu'))(x)
    x = Flatten()(x)
    x = Dense(units=1, activation=('sigmoid'))(x)
    md = Model(inputs, x)
    md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    md.fit(X_train, y_train, batch_size=32, epochs=100, verbose=True)
    # Making the Confusion Matrix
    y_pred = md.predict(X_test)
    y_pred = (y_pred > 0.5)
    #print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    cm = confusion_matrix(y_test==1, y_pred)
    print(cm)
    y_ravel = md.predict(X_test).ravel()
    fpr[i], tpr[i], _ = roc_curve(y_test==1, y_ravel)
    roc_auc[i] = auc(fpr[i], tpr[i])
a = np.empty(0)
for i in range(10):
    a=np.append(a,roc_auc[i])

np.mean(a)
st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
#accuracy_score(y_test, y_pred)
#record.append(cm)
#colors = ['navy', 'gray', 'lightcoral', 'gold', 'chartreuse', 'dodgerblue', 'magenta', 'aqua', 'darkorange', 'cornflowerblue']
plt.figure(figsize=(5.5,4))
lw = 2
for i in range(10):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])

plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('FCL on Age & Sex')
plt.legend(loc="lower right")
plt.savefig('FCL_Age_&_Sex.png')
'''
tt=0
for i in range(10):
    tt = roc_auc[i]+tt
tt/10
'''
