import numpy as np
import pandas as pd
import scipy.stats as st
import tensorflow as tf
from sklearn import preprocessing
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
AHF_data = pd.read_csv('AHF.csv', header=None)
normal_data = pd.read_csv('normal.csv', header=None)
X = AHF_data.iloc[:, 0:-1].values
y = AHF_data.iloc[:, -1].values
X = np.append(X, normal_data.iloc[:,0:-1].values, axis=0)
y = np.append(y, normal_data.iloc[:,-1].values, axis=0)
minmax = preprocessing.MinMaxScaler()
X = minmax.fit_transform(X)
X_100=np.empty([2908,0])
for i in range(100):
	X_100=np.append(X_100, X[:,int(i*10)].reshape(2908,1), axis=1)

from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Flatten, LSTM, BatchNormalization, Dropout, GRU, Concatenate, Activation
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from random import randint

tpr = dict()
fpr = dict()
roc_auc = dict()
n = 2908

for i in range(10):
	X_train, X_test, y_train, y_test = train_test_split(X_100, y, test_size = 0.2, random_state = i+1)
	X_train = X_train.reshape(len(X_train),100,1).astype('float64')
	y_train = y_train.reshape(len(y_train)).astype('int')
	X_test = X_test.reshape(len(X_test),100,1).astype('float64')
	y_test = y_test.reshape(len(y_test)).astype('int')
	inputs = Input(shape=(100,1))
	x = Dense(units=100, activation=('relu'))(inputs)
	x = Flatten()(x)
	x = Dense(units=1, activation=('sigmoid'))(x)
	md = Model(inputs, x)
	md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
	md.fit(X_train, y_train, batch_size=32, epochs=100, verbose=True)
	y_pred = md.predict(X_test)
	y_pred = (y_pred > 0.5)
	cm = confusion_matrix(y_test, y_pred)
	print(cm)
	y_ravel = md.predict(X_test).ravel()
	fpr[i], tpr[i], _ = roc_curve(y_test, y_ravel)
	roc_auc[i] = auc(fpr[i], tpr[i])

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
plt.title('Perceptron trained on PWs')
plt.legend(loc="lower right")
plt.savefig('Perceptron PWs.png')
    
