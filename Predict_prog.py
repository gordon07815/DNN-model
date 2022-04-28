# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:00:27 2021

@author: Gordon Huang
"""

import pandas as pd
import numpy as np
import math
import re
ahf_prog = pd.read_excel('AHF_outcome.xlsx')
ev = [0]*431
eve = [ev]*4

for i in range(431):
    if(type(ahf_prog['Event 1'].astype('string')[i]) != pd._libs.missing.NAType):
        eve[0][i] = 1
    if(type(ahf_prog['Event 2'].astype('string')[i]) != pd._libs.missing.NAType):
        eve[1][i] = 1
    if(type(ahf_prog['Event 3'].astype('string')[i]) != pd._libs.missing.NAType):
        eve[2][i] = 1
    if(ahf_prog['Mortality'][i] == True):
        eve[3][i] = 1

    
ahf_data = pd.read_excel("V2 lab.xlsx")
age_ahf = []
for i in range(433):
    if(str(ahf_data.檢查日[i]).find("NaT") >= 0):
        if(str(ahf_data.出生日[i]).find("NaT") >= 0):
            age_ahf = np.append(age_ahf, "60")
        else:
            age_ahf = np.append(age_ahf, int(2007) - int(str(ahf_data.出生日[i])[:4]))
    elif(str(ahf_data.出生日[i]).find("NaT") >= 0):
        age_ahf = np.append(age_ahf, "60")
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
AHF_data = pd.read_csv('AHF_noted.csv', header=None)
AHF_data = pd.concat([AHF_data[range(1000)], AHF_data[1002]],axis=1)
AHF_data= AHF_data.rename(columns={1002:1000})
array = np.empty([0,1004])
for i in range(len(AHF_data)):
    if(re.search(r'AHF\d*',AHF_data[0][i])):
        a = re.search(r'AHF\d*',AHF_data[0][i])
        n = a.group(0)[3:6]
        if(ahf_data['編號'].str.contains(a.group(0)).any()):
            b = np.append(AHF_data.iloc[i,:998], age_ahf[pd.Index(ahf_data['編號']).get_loc(a.group(0))])[:1000].reshape(1,1000)
            b = np.append(b, np.array(eve)[:,int(n)-1])
            array = np.append(array,[b],axis=0)

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import Input, Model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, LSTM, BatchNormalization, Dropout, MaxoutDense, GRU, Concatenate, Activation, GlobalAveragePooling1D, Lambda
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from random import randint
tpr = dict()
fpr = dict()
roc_auc = dict()

X_train, X_test, y_train, y_test = train_test_split(array[:,1:-1], array[:,-1], test_size = 0.2, random_state = 1)
X_train = X_train.reshape(len(X_train), 1002, 1).astype('float64')
X_test = X_test.reshape(len(X_test), 1002, 1).astype('float64')
y_train = y_train.reshape(len(y_train),).astype('int')
y_test = y_test.reshape(len(y_test),).astype('int')

def get(x):
	return x[:,:-5]
def left(x):
	return x[:,-3:]
input = Input(shape=(1002,1))
x1 = Lambda(get, (997,1))(input)
x2 = Lambda(get, (997,1))(input)
x3 = Lambda(get, (997,1))(input)
x4 = Lambda(get, (997,1))(input)
x5 = Lambda(get, (997,1))(input)
x6 = Lambda(get, (997,1))(input)
x7 = Lambda(get, (997,1))(input)
x8 = Lambda(get, (997,1))(input)
x9 = Lambda(get, (997,1))(input)
x10 = Lambda(get, (997,1))(input)
x11 = Lambda(get, (997,1))(input)
x12 = Lambda(get, (997,1))(input)
x_as = Lambda(left, (2,1))(input)
x1 = Conv1D(256, 2, activation=('relu'))(input)
x2 = Conv1D(256, 3, activation=('relu'))(input)
x3 = Conv1D(128, 2, activation=('relu'))(input)
x4 = Conv1D(128, 3, activation=('relu'))(input)
x5 = Conv1D(256, 4, activation=('relu'))(input)
x6 = Conv1D(128, 4, activation=('relu'))(input)
x7 = Conv1D(64, 2, activation=('relu'))(input)
x8 = Conv1D(64, 3, activation=('relu'))(input)
x9 = Conv1D(64, 4, activation=('relu'))(input)
x10 = Conv1D(32, 2, activation=('relu'))(input)
x11 = Conv1D(32, 3, activation=('relu'))(input)
x12 = Conv1D(32, 4, activation=('relu'))(input)
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
x_as = Dense(units=64, activation=('relu'))(x_as)
x = Concatenate(axis=1)([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x_as])
x = Dense(units=8, activation=('relu'))(x)
x = Flatten()(x)
x = Dense(units=1, activation=('sigmoid'))(x)
md = Model(input, x)
stop = EarlyStopping(monitor='loss', patience=9)
md.compile(optimizer='AdaGrad', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100, verbose=True, callbacks=[stop])

y_pred = md.predict([X_test,])
y_pred = (y_pred > 0.5)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
cm = confusion_matrix(y_test, y_pred)
print(cm)
y_ravel = md.predict(X_test).ravel()
fpr, tpr, _ = roc_curve(y_test, y_ravel)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC_prognosis_no_events.png')
