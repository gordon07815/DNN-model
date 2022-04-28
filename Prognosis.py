#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 16:28:01 2022

@author: gordon07815
"""
from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.under_sampling import NearMiss
import numpy as np
import math
import re
AHF_data = pd.read_csv('AHF_noted.csv', header=None)
ahf_prog = pd.read_excel('AHF_outcome.xlsx')
ahf_prog['V2日期'][393]='2018/1/1'
eve = pd.DataFrame([[0]*431]*6)
for i in range(431):
    if(type(ahf_prog['Event 1'].astype('string')[i]) != pd._libs.missing.NAType):
        eve.iloc[0][i] = 1
    if(type(ahf_prog['Event 2'].astype('string')[i]) != pd._libs.missing.NAType):
        eve.iloc[1][i] = 1
    if(type(ahf_prog['Event 3'].astype('string')[i]) != pd._libs.missing.NAType):
        eve.iloc[2][i] = 1
    if(type(ahf_prog['Event 4'].astype('string')[i]) != pd._libs.missing.NAType):
        eve.iloc[3][i] = 1
    if(str(ahf_prog['Mortality date'][i])!='nan'):
        eve.iloc[5][i] = 1
        a = re.match(r'(\d+)\D*(\d*)\D*(\d*)', str(ahf_prog['Mortality date'][i]))
        b = re.match(r'(\d+)\D*(\d*)\D*(\d*)', str(ahf_prog['V2日期'][i]))
        if(a.group(2)==''):
            eve.iloc[4][i] = (date(int(a.group(1)), 12, 31) - date(int(b.group(1)), int(b.group(2)), int(b.group(3)))).days
        elif(a.group(3)==''):
            eve.iloc[4][i] = (date(int(a.group(1)), int(a.group(2)), 31) - date(int(b.group(1)), int(b.group(2)), int(b.group(3)))).days
        else:
            eve.iloc[4][i] = (date(int(a.group(1)), int(a.group(2)), int(a.group(3))) - date(int(b.group(1)), int(b.group(2)), int(b.group(3)))).days
    else:
        b = re.match(r'(\d+)\D*(\d*)\D*(\d*)', str(ahf_prog['V2日期'][i]))
        eve.iloc[4][i] = (date(2021, 12, 10) - date(int(b.group(1)), int(b.group(2)), int(b.group(3)))).days
array = np.empty([0,1001])
ahf_data = pd.read_excel("V2 lab_modified.xlsx")
for i in range(len(AHF_data)):
    if(re.search(r'AHF\d*V2',AHF_data[0][i])):
        a = re.search(r'AHF(\d*)',AHF_data[0][i])
        n = a.group(1)
        if(ahf_data['編號'].str.contains(a.group(0)).any()):
            b = np.append(AHF_data.iloc[i,range(1,1001)], np.array(eve)[4:,int(n)-1])[:1001].reshape(1,1001)
            array = np.append(array,b,axis=0)
array = np.array(array)

X, y = NearMiss().fit_resample(array[:, 0:-1].astype('float64'), 1*(array[:, -1].astype('int')>730).astype('int'))
tpr = dict()
fpr = dict()
roc_auc = dict()
#y_mod = np.asarray([0]*500+[1]*433).astype('int')
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Embedding, Dense, MaxPool1D, Flatten, LSTM, BatchNormalization, Dropout, GRU, Concatenate, Activation
from sklearn.metrics import  confusion_matrix, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay
import seaborn as sns
#python -m pip install -I matplotlib
import matplotlib.pyplot as plt
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = i+1)
    ''' X_train = np.append(X[0:int(i*n/10),:],X[int((i+1)*n/10):,:],axis=0)
    X_test = X[int(i*n/10):int((i+1)*n/10),:]
    y_train = np.append(y[0:int(i*n/10)],y[int((i+1)*n/10):],axis=0)
    y_test = y[int(i*n/10):int((i+1)*n/10)]'''
    X_train = X_train.reshape(len(X_train),1000,1).astype('float64')
    X_test = X_test.reshape(len(X_test),1000,1).astype('float64')
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    inputs = Input(shape=(1000,1))
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
    x1 = Dense(units=64, activation=('relu'))(x1)
    x2 = Dense(units=64, activation=('relu'))(x2)
    x3 = Dense(units=64, activation=('relu'))(x3)
    x4 = Dense(units=64, activation=('relu'))(x4)
    x5 = Dense(units=64, activation=('relu'))(x5)
    x6 = Dense(units=64, activation=('relu'))(x6)
    x7 = Dense(units=64, activation=('relu'))(x7)
    x8 = Dense(units=64, activation=('relu'))(x8)
    x9 = Dense(units=64, activation=('relu'))(x9)
    x10 = Dense(units=64, activation=('relu'))(x10)
    x11 = Dense(units=64, activation=('relu'))(x11)
    x12 = Dense(units=64, activation=('relu'))(x12)
    x = Concatenate(axis=1)([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12])
        #x = Dense(units=512, activation=('relu'))(inputs)
        
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
plt.figure()
lw = 2
for i in range(10):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])

plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('1-year survival prediction')
plt.legend(loc="lower right")
plt.savefig('2-year.png')

tt=0.0
for i in range(10):
    tt=roc_auc[i]+tt
tt/10