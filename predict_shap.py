# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
tf.__version__
from tqdm import tqdm
import shap
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Part 1 - Data Preprocessing

# Importing the dataset
AHF_data = pd.read_csv('AHF_noted.csv', header=None)
normal_data = pd.read_csv('normal_noted.csv', header=None)
X = AHF_data.iloc[:, :1001].values
y = AHF_data.iloc[:, -1].values
X = np.append(X, normal_data.iloc[:,0:1001].values, axis=0)
y = np.append(y, normal_data.iloc[:,-1].values, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
name = X_test[:100,0]

# Encoding categorical data
# Label Encoding the "Gender" column
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#X[:, 2] = le.fit_transform(X[:, 2])
#print(X)
# One Hot Encoding the "Geography" column
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
#X = np.array(ct.fit_transform(X))
#print(X)

# Splitting the dataset into the Training set and Test set
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Embedding, Dense, MaxPool1D, Flatten, LSTM, BatchNormalization, Dropout, GRU, Concatenate, Activation
from sklearn.metrics import  confusion_matrix, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay
#python -m pip install -I matplotlib
import matplotlib.pyplot as plt
from random import randint
from tensorflow.keras.preprocessing import sequence
#from itertools import cycle

#record = []
tpr = dict()
fpr = dict()
roc_auc = dict()
n = 2908
#for i in range(10):
X_100=np.empty([2908,0])
for i in range(100):
    X_100=np.append(X_100, X[:,int(i*10)+1].reshape(2908,1), axis=1)

minmax = preprocessing.MinMaxScaler()
X_100 = minmax.fit_transform(X_100.T).T

#X_train, X_test, y_train, y_test = train_test_split(X_100, y, test_size = 0.2, random_state = 1)
#'AHF361V3', 3411001.0, 2209303.0, 'AHF281V3', 2204801.0, 'AHF369V4', 'AHF326V6', 2413402.0, 'AHF204V1', 'AHF420V08']
''' X_train = np.append(X[0:int(i*n/10),:],X[int((i+1)*n/10):,:],axis=0)
X_test = X[int(i*n/10):int((i+1)*n/10),:]
y_train = np.append(y[0:int(i*n/10)],y[int((i+1)*n/10):],axis=0)
y_test = y[int(i*n/10):int((i+1)*n/10)]'''
X_train = X_100.reshape(len(X_100),100,1).astype('float64')
#X_test = X_test.reshape(len(X_test),100,1).astype('float64')
y_train = y.astype('int')
#y_test= y_test.astype('int')

inputs = Input(shape=(100,1))
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
x1 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x1)
x2 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x2)
x3 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x3)
x4 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x4)
x5 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x5)
x6 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x6)
x7 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x7)
x8 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x8)
x9 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x9)
x10 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x10)
x11 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x11)
x12 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1, training=True)(x12)
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
x = Dense(units=8, activation=('relu'))(x)
x = Flatten()(x)
x = Dense(units=1, activation=('sigmoid'))(x)
md = Model(inputs, x)
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100, verbose=True)
'''
plt.close('all')
cm=confusion_matrix(y_test==1, md.predict(X_test)>0.5)
ax = sns.heatmap(cm, annot=True, fmt='1',cmap='Blues')
ax.set_title('Conv1D trained on 8:2 PWs');
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Actual Values');
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
plt.savefig('CM of Conv1D trained on 8:2 PWs.png')
#https://shap-lrjball.readthedocs.io/en/latest/example_notebooks/deep_explainer/Keras%20LSTM%20for%20IMDB%20Sentiment%20Classification.html
#https://www.yourdatateacher.com/2021/05/17/how-to-explain-neural-networks-using-shap/
#X_train = shap.kmeans(X_train, 2326)
'''
#explainer = shap.KernelExplainer(md, X_train.reshape(len(X_train), 1000))

#shap.explainers._deep.deep_tf.op_handlers['AddV2'] = shap.explainers._deep.deep_tf.passthrough
#explainer = shap.DeepExplainer(md, X_train)
X_train, X_test, y_train, y_test = train_test_split(X_100, y, test_size = 0.2, random_state = 1)
explainer = shap.KernelExplainer(md, X_train[:100].reshape(100, 100))
shap_values = explainer.shap_values(X_test[100:200].reshape(100, 100), nsample=2000) #nsamples = 2 * X.shape[1] + 2048 = 2248
#Using 2326 background data samples could cause slower run times. 
#Consider using shap.sample(data, K) or shap.kmeans(data, K) to summarize the background as K samples.
fig, axs=plt.subplots(2,5, figsize=(40,8))
for i in range(2):
    for j in range(5):
        axs[i,j].plot(X_test[i*5+j+5,:])
        axs[i,j].set_title(name[i*5+j])

plt.savefig('plots.png')
plt.close('all')

w=np.empty(0)
for i in range(100):
    w=np.append(w, np.array(str(i)))

shap.initjs()
for i in range(100):
    plt.figure()
    shap.force_plot(explainer.expected_value[0], shap_values[0][i], np.around(X_test[i], 3), feature_names=w, matplotlib=True)
    plt.savefig("%s_force_plot.png"%name[i])


plt.figure()
shap.summary_plot(shap_values, w)
plt.savefig('summary.png')

'''for i in range(10):
    plt.figure()
    shap.force_plot(explainer.expected_value, shap_values[0][i], X_test[i], feature_names=w, matplotlib=True)
    plt.show()'''

'''
113.90 is the predicted value. The base value is the average value of the target variable across all the records. 
Each stripe shows the impact of its feature in pushing the value of the target variable farther or closer to the base value.
Red stripes show that their features push the value towards higher values. 
Blue stripes show that their features push the value towards lower values. 
The wider a stripe, the higher (in absolute value) the contribution. 
The sum of these contributions pushes the value of the target variable from the vase value to the final, predicted value.
'''


for i in range(10):
    plt.plot(X_test[i+5,1:100])
    plt.title(name[i])
    plt.savefig('%s.png'%name[i])
    plt.close()

plt.close('all')
a=np.empty(100)
b=np.empty(100)
for i in range(100):
    a=a+shap_values[0][i]
    b=b+abs(shap_values[0][i])

plt.plot(a)
plt.savefig('Shap values.png')
plt.close()
plt.plot(b)
plt.savefig('Absolute shap values.png')

for i in range(10):
    plt.figure()
    plt.plot(range(100),shap_values[0][i])
    plt.title(name[i])
    plt.savefig('%s.png'%name[i])
    plt.figure()
    plt.plot(range(100),abs(shap_values[0][i]))
    plt.title("%s_absolute"%name[i])
    plt.savefig("%s_absolute.png"%name[i])


    