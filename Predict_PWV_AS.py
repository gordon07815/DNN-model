import pandas as pd
import numpy as np
import scipy.stats as st
import math
import re
from sklearn import preprocessing

ahf_data = pd.read_excel("V2 lab.xlsx")
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

AHF_data = pd.read_csv('AHF_noted.csv', header=None)
AHF_data = pd.concat([AHF_data[0],AHF_data[np.arange(1,1001,10)]],axis=1)
AHF_data= np.array(AHF_data)
array = np.empty([1,103])
for i in range(len(AHF_data)):
    if(re.search(r'AHF\d*',AHF_data[i][0])):
        a = re.search(r'AHF\d*',AHF_data[i][0])
        if(ahf_data['編號'].str.contains(a.group(0)).any()):
            b = np.append(AHF_data[i,1:101], age_ahf[pd.Index(ahf_data['編號']).get_loc(a.group(0))])[:103].reshape(1,103)
            array = np.append(array,b,axis=0)

normal_data = pd.read_csv('normal_noted.csv', header=None)
normal_data = pd.concat([normal_data[0], normal_data[np.arange(1,1000,10)]],axis=1)
for i in range(len(normal_data)):
    if(Normal_age_sex.ID.astype(str).str.contains(normal_data.iloc[i,0].astype(str)).any()):
        j = np.where(Normal_age_sex.ID==normal_data.iloc[i,0])[0][0]
        if(i==0):
            normal_data_A_S = np.append(normal_data.iloc[i,1:], 
            np.append(Normal_age_sex.N_Age[j],
            np.append(Normal_age_sex.M_sex[j], 0))).reshape(1,103)
        else:    
            normal_data_A_S = np.concatenate((
            normal_data_A_S,
            np.append(normal_data.iloc[i,1:], 
            np.append(Normal_age_sex.N_Age[j],
            np.append(Normal_age_sex.M_sex[j], 0))).reshape(1,103)), axis=0)

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Flatten, LSTM, BatchNormalization, Dropout, GRU, Concatenate, Activation, GlobalAveragePooling1D, Lambda
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from random import randint
'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
'''
tpr = dict()
fpr = dict()
roc_auc = dict()
n = 2758

#https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/

def get(x):
	return x[:,:-2]

def left(x):
	return x[:,-2:]

array = np.append(array[1:,:], normal_data_A_S[:,:], axis=0)
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(array[:,:-1], array[:,-1], test_size = 0.2, random_state = i+1)
    X_train = X_train.reshape(len(X_train), 102, 1).astype('float64')
    X_test = X_test.reshape(len(X_test), 102, 1).astype('float64')
    y_train = y_train.reshape(len(y_train),).astype('int')
    y_test = y_test.reshape(len(y_test),).astype('int')
    inputs = Input(shape=(102,1))
    x1 = Lambda(get, (100,1))(inputs)
    x2 = Lambda(get, (100,1))(inputs)
    x3 = Lambda(get, (100,1))(inputs)
    x4 = Lambda(get, (100,1))(inputs)
    x5 = Lambda(get, (100,1))(inputs)
    x6 = Lambda(get, (100,1))(inputs)
    x7 = Lambda(get, (100,1))(inputs)
    x8 = Lambda(get, (100,1))(inputs)
    x9 = Lambda(get, (100,1))(inputs)
    x10 = Lambda(get, (100,1))(inputs)
    x11 = Lambda(get, (100,1))(inputs)
    x12 = Lambda(get, (100,1))(inputs)
    x_as = Lambda(left, (2,1))(inputs)
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
    md = Model(inputs, x)
    md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    md.fit(X_train, y_train, batch_size=32, epochs=100, verbose=True)
    y_pred = md.predict([X_test,])
    y_pred = (y_pred > 0.5)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    y_ravel = md.predict(X_test).ravel()
    fpr[i], tpr[i], _ = roc_curve(y_test, y_ravel)
    roc_auc[i] = auc(fpr[i], tpr[i])
#colors = ['navy', 'gray', 'lightcoral', 'gold', 'chartreuse', 'dodgerblue', 'magenta', 'aqua', 'darkorange', 'cornflowerblue']
a = np.empty(0)
for i in range(10):
    a=np.append(a,roc_auc[i])

np.mean(a)
st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))

plt.figure()
lw = 2
for i in range(10):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])

plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Conv1D trained on PWs with Age & Sex')
plt.legend(loc="lower right")
plt.savefig('Conv1D PWs with Age & Sex.png')
