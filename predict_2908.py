# Artificial Neural Network

# Importing the libraries
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
#minmax = preprocessing.MinMaxScaler()
#X = minmax.fit_transform(X)
X_100=np.empty([2908,0])
for i in range(100):
    X_100=np.append(X_100, X[:,int(i*10)].reshape(2908,1), axis=1)

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
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Flatten, LSTM, BatchNormalization, Dropout, GRU, Concatenate, Activation
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, matthews_corrcoef
import matplotlib.pyplot as plt
from random import randint
#from itertools import cycle

#record = []

def Conv(k, f, x):
    y = Conv1D(k, f, activation=('relu'))(x)
    y = MaxPool1D()(y)
    y = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(y)
    return y

n = 2908
dnn = np.empty([4,2,0])
for k in range(1,6):
    for j in range(1,10):
        sen = np.empty(0)
        spe = np.empty(0)
        mcc = np.empty(0)
        roc_auc = np.empty(0)
        X_stack = X_100[np.arange(0,len(X_100),k)]
        y_stack = y[np.arange(0,len(X_100),k)]
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X_stack, y_stack, test_size = (10-j)/10, random_state = i+1)
            X_train = X_train.reshape(len(X_train),100,1).astype('float64')
            y_train = y_train.reshape(len(y_train)).astype('int')
            X_test = X_test.reshape(len(X_test),100,1).astype('float64')
            y_test = y_test.reshape(len(y_test)).astype('int')
            inputs = Input(shape=(100,1))
            x1 = Conv(128, 2, inputs)
            x2 = Conv(128, 3, inputs)
            x3 = Conv(256, 2, inputs)
            x4 = Conv(256, 3, inputs)
            x5 = Conv(128, 4, inputs)
            x6 = Conv(256, 4, inputs)
            x7 = Conv(32, 2, inputs)
            x8 = Conv(32, 3, inputs)
            x9 = Conv(32, 4, inputs)
            x10 = Conv(64, 2, inputs)
            x11 = Conv(64, 3, inputs)
            x12 = Conv(64, 4, inputs)
            x1 = Conv(128, 2, x1)
            x2 = Conv(128, 3, x2)
            x3 = Conv(256, 2, x3)
            x4 = Conv(256, 3, x4)
            x5 = Conv(128, 4, x5)
            x6 = Conv(256, 4, x6)
            x7 = Conv(32, 2, x7)
            x8 = Conv(32, 3, x8)
            x9 = Conv(32, 4, x9)
            x10 = Conv(64, 2, x10)
            x11 = Conv(64, 3, x11)
            x12 = Conv(64, 4, x12)
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
            y_ravel = md.predict(X_test).ravel()
            y_pred = (y_ravel > 0.5)
            cm = confusion_matrix(y_test, y_pred)
            print(cm)
            fpr, tpr, _ = roc_curve(y_test, y_ravel)
            roc_auc = np.append(roc_auc, auc(fpr, tpr))
            spe = np.append(spe, cm[0,0]/(cm[0,0]+cm[0,1]))
            sen = np.append(sen, cm[1,1]/(cm[1,1]+cm[1,0]))
            mcc = np.append(mcc, matthews_corrcoef(y_test, y_pred))
        sen = [np.mean(sen), st.t.interval(0.95, len(sen)-1, loc=np.mean(sen), scale=st.sem(sen))[1]-np.mean(sen)]
        spe = [np.mean(spe), st.t.interval(0.95, len(spe)-1, loc=np.mean(spe), scale=st.sem(spe))[1]-np.mean(spe)]
        mcc = [np.mean(mcc), st.t.interval(0.95, len(mcc)-1, loc=np.mean(mcc), scale=st.sem(mcc))[1]-np.mean(mcc)]
        roc_auc = [np.mean(roc_auc), st.t.interval(0.95, len(roc_auc)-1, loc=np.mean(roc_auc), scale=st.sem(roc_auc))[1]-np.mean(roc_auc)]
        dnn = np.append(dnn, np.array([sen, spe, mcc, roc_auc]).reshape(4,2,1), axis=2)

a = ['Sensitivity', 'Specificity', 'Matthew correlation coefficient', 'Area under ROC']
for i in range(4):
	for j in range(5):
		plt.errorbar(range(1,10), dnn[i,0,(4-j)*9:(4-j+1)*9], dnn[i,1,(4-j)*9:(4-j+1)*9], fmt='-o', elinewidth=2, capsize=4, label='DNN model under 1/%d data'%(5-j))
		#plt.ylim([0.65,0.91])
	plt.xticks(range(1,10),['1:9','2:8','3:7','4:6','5:5','6:4','7:3','8:2','9:1'])
	plt.title('%s'%a[i])
	plt.legend()
	plt.savefig('%s.png'%a[i])
	plt.close()

lim = [[0.5, 0.92], [0.4, 0.95], [0.25, 0.8], [0.7, 1.0]]
for i in range(4):
    for j in range(9):
        plt.errorbar(range(1,6), dnn[i,0,(4-np.arange(5))*9+j], dnn[i,1,(4-np.arange(5))*9+j], fmt='-o', elinewidth=2, capsize=4, label='Train-Test ratio of %d:%d'%(j+1,9-j))
        plt.ylim(lim[i])
    plt.xticks(range(1,6),['1/%d data' % (6-k) for k in range(1,6)])
    plt.title('%s'%a[i])
    plt.legend()
    plt.savefig('%s.png'%a[i])
    plt.close()

lim = [[0.5, 0.92], [0.5, 1.0], [0.25, 0.8], [0.7, 1.0]]
for i in range(4):
    for j in range(4):
        plt.errorbar(range(1,6), dnn[i,0,(4-np.arange(5))*9+j+5], dnn[i,1,(4-np.arange(5))*9+j+5], fmt='-o', elinewidth=2, capsize=4, label='Train-Test ratio of %d:%d'%(j+6,4-j))
        plt.ylim(lim[i])
    plt.xticks(range(1,6),['1/%d data' % (6-k) for k in range(1,6)])
    plt.title('%s'%a[i])
    plt.legend()
    plt.savefig('%s.png'%a[i])
    plt.close()



'''
Sensitivity: [[[0.72080767, 0.75277935, 0.79037715, 0.84389053, 0.85653037,
         0.83633491, 0.82382212, 0.83677669, 0.85191866],
        [0.05647963, 0.04696276, 0.05941303, 0.04259481, 0.05029525,
         0.03670106, 0.03147352, 0.02574078, 0.054863  ]],

Specificity: [[0.80614561, 0.82014815, 0.8213322 , 0.79861849, 0.74786905,
         0.80868336, 0.86476892, 0.82693763, 0.80011678],
        [0.05270406, 0.05839959, 0.06391714, 0.07706352, 0.15240716,
         0.06547745, 0.02845291, 0.04362186, 0.08662938]],

MCC: [[0.53263941, 0.57813595, 0.61754442, 0.65006189, 0.62067349,
         0.64985497, 0.68893686, 0.66557579, 0.6613496 ],
        [0.01479813, 0.01520029, 0.03921513, 0.04121381, 0.08903856,
         0.03888705, 0.02823795, 0.02966342, 0.05323821]],

AUROC: [[0.83296286, 0.86173598, 0.88700441, 0.9051494 , 0.90132614,
         0.90623848, 0.91587866, 0.90929115, 0.91094324],
        [0.01115842, 0.00751223, 0.01240429, 0.01216628, 0.01380572,
         0.01425184, 0.0139714 , 0.01521237, 0.01624161]]]
'''
#accuracy_score(y_test, y_pred)
#record.append(cm)
#colors = ['navy', 'gray', 'lightcoral', 'gold', 'chartreuse', 'dodgerblue', 'magenta', 'aqua', 'darkorange', 'cornflowerblue']
plt.figure()
lw = 2
for i in range(10):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])

plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Conv1D trained with 8:2 PWs')
plt.legend(loc="lower right")
plt.savefig('Conv1D PWs 8:2.png')

'''
a = np.empty(0)
for i in range(10):
    a=np.append(a,mcc[i])

print("%.3f"%np.mean(a))
print("[%.3f, %.3f]"%st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a)))'''

a=[0.863,0.906,0.919,0.904]
b=[0.011,0.015,0.01,0.024]

plt.subplots(figsize=(6,4))
plt.errorbar(range(1,5), a, b, fmt='-o', elinewidth=2, capsize=4)
plt.ylim([0.8,1.0])
plt.xticks(range(1,5),['2:8','4:6','6:4','8:2'])
plt.title('Area under ROC curve')
plt.savefig('AUROC.png')


'''
Sensitivity:
    2:8>>0.782[0.735,0.829]
    4:6>>0.840[0.802,0.879]
    6:4>>0.830[0.802,0.858]
    8:2>>0.918[0.878,0.958]

Specificity:
    2:8>>0.793[0.736,0.849]
    4:6>>0.809[0.732,0.885]
    6:4>>0.860[0.821,0.898]
    8:2>>0.581[0.350,0.813]

MCC:
    2:8>>0.579[0.551,0.606]
    4:6>>0.655[0.612,0.699]
    6:4>>0.690[0.668,0.712]
    8:2>>0.526[0.346,0.707]

AUROC:
    2:8>>0.863[0.852,0.874]
    4:6>>0.906[0.891,0.921]
    6:4>>0.919[0.909,0.928]
    8:2>>0.904[0.880,0.927]


from tqdm import tqdm
import shap

def f(X):
    return md.predict([X[:,i] for i in range(X.shape[1])]).flatten()
explainer = shap.Explainer(md, X_train)
shap_values = explainer.shap_values(X_test[:10])
shap.force_plot(explainer.expected_value, shap_values, X[299,:])
shap_values50 = explainer.shap_values(X[280:330,:], nsamples=500)
shap.force_plot(explainer.expected_value, shap_values50, X[280:330,:])
'''
# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# ANN: - loss: 0.6929 - accuracy: 0.5125
'''
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='softmax'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
'''
#Conv1D(2913)
    
'''

#method 1: - loss: 0.6929 - accuracy: 0.5125
md = Sequential()
md.add(Conv1D(64,2,activation = 'relu', input_shape=(100,1)))
md.add(MaxPool1D())
md.add(Flatten())
md.add(Dense(units=32, activation=('relu')))
md.add(Dense(units=1, activation='sigmoid'))
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.summary()
md.fit(X_train, y_train, batch_size=32, epochs=100)



#method 2: - loss: 0.6527 - accuracy: 0.6062
md = Sequential()
md.add(Conv1D(64,2,activation = 'relu', padding='same', input_shape=(100,1)))
md.add(MaxPool1D())
md.add(LSTM(64))
md.add(Dense(units=32, activation=('linear')))
md.add(Dense(units=1, activation='sigmoid'))
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100)


#method 3: - loss: 0.3077 - accuracy: 0.8500; [[18  4][ 8 10]](0.7)
md = Sequential()
md.add(Conv1D(256,3,activation = 'relu', input_shape=(1000,1)))
md.add(MaxPool1D())
md.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)) 
md.add(Conv1D(256,3,activation = 'relu', input_shape=(1000,1)))
md.add(MaxPool1D())     
md.add(Flatten())
md.add(Dense(units=64, activation=('relu')))
md.add(Dense(units=1, activation='sigmoid'))
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100)

#method 4: - loss: 0.4742 - accuracy: 0.7312; [[18  4][12  6]](0.6)
md = Sequential()
md.add(Conv1D(256,3,activation = 'relu', input_shape=(1000,1)))
md.add(MaxPool1D())
md.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)) 
md.add(Conv1D(256,3,activation = 'relu', input_shape=(1000,1)))
md.add(MaxPool1D())     
md.add(Flatten())
md.add(Dropout(0.5))
md.add(Dense(units=64, activation=('relu')))
md.add(Dropout(0.5))  
md.add(Dense(units=1, activation='sigmoid'))
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100)

#method 5: - loss: 0.6057 - accuracy: 0.6313; [[21  1][13  5]](0.65)
md = Sequential()
md.add(Conv1D(256,3,activation = 'relu', input_shape=(1000,1)))
md.add(MaxPool1D())
md.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)) 
md.add(Conv1D(256,3,activation = 'relu', input_shape=(1000,1)))
md.add(MaxPool1D())     
md.add(Flatten())
md.add(Dropout(0.5))
md.add(MaxoutDense(512, nb_feature=4))
md.add(Dense(units=64, activation=('relu')))
md.add(Dropout(0.5))  
md.add(MaxoutDense(512, nb_feature=4))  
md.add(Dense(units=1, activation='sigmoid'))
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100)

#method 6: - loss: 7.4750 - accuracy: 0.5125
md = Sequential()
md.add(Conv1D(64,2,activation = 'relu', input_shape=(1000,1)))
md.add(MaxPool1D())     
md.add(Conv1D(64,2,activation = 'relu', input_shape=(1000,1)))
md.add(MaxPool1D())     
md.add(Conv1D(64,2,activation = 'relu', input_shape=(1000,1)))
md.add(MaxPool1D())   
md.add(GRU(64,return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
md.add(Flatten())
md.add(Dense(units=32, activation=('relu')))
md.add(Dense(units=1, activation='softmax'))
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100)


#method 7: - loss: 0.6695 - accuracy: 0.5875; [[14  8][11  7]](0.525)
md = Sequential()
md.add(GRU(128, input_shape=(1000,1), return_sequences=True))
md.add(GRU(64))
md.add(Dropout(0.5))
md.add(Dense(units=32, activation=('relu')))
md.add(Dense(units=1, activation='sigmoid'))
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100)


#method 8: - loss: 0.6930 - accuracy: 0.5125; [[ 0 22][ 0 18]](0.45)
md = Sequential()
md.add(Conv1D(64,2,activation = 'relu'))
md.add(MaxPool1D())
md.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)) 
md.add(Flatten())
md.add(Dense(units=32, activation=('relu')))
md.add(Dense(units=8, activation=('relu')))
md.add(Dense(units=1, activation='sigmoid'))
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100)


#method 9: - loss: 0.4380 - accuracy: 0.7750; [[20  2][11  7]](0.675)
inputs = Input(shape=(1000,1))
x1 = Conv1D(64, 2, activation=('relu'))(inputs)
x2 = Conv1D(64, 3, activation=('relu'))(inputs)
x3 = Conv1D(64, 4, activation=('relu'))(inputs)
x1 = MaxPool1D()(x1)
x2 = MaxPool1D()(x2)
x3 = MaxPool1D()(x3)
x1 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x1)
x2 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x2)
x3 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x3)
x1 = Dense(units=32, activation=('relu'))(x1)
x2 = Dense(units=32, activation=('relu'))(x2)
x3 = Dense(units=32, activation=('relu'))(x3)
x = Concatenate(axis=1)([x1,x2,x3])
x = Dense(units=8, activation=('relu'))(x)
x = Flatten()(x)
x = Dense(units=1, activation=('sigmoid'))(x)
md = Model(inputs, x)
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100)

#method 10: - loss: 0.4430 - accuracy: 0.7937; [[ 1 21][ 0 18]](0.475)
md = Sequential()
md.add(Conv1D(256,3,activation = 'relu', input_shape=(1000,1)))
md.add(Conv1D(256,3,activation = 'relu', input_shape=(1000,1)))    
md.add(Flatten())
md.add(Dense(units=64, activation=('relu')))
md.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)) 
md.add(Dropout(0.2))
md.add(Dense(units=32, activation=('relu')))
md.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)) 
md.add(Dense(units=1, activation='sigmoid'))
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100)

#method 11: - loss: 0.4015 - accuracy: 0.8188; [[18  4][11  7]](0.625)
inputs = Input(shape=(1000,1))
x1 = Conv1D(256, 2, activation=('relu'))(inputs)
x2 = Conv1D(256, 3, activation=('relu'))(inputs)
x3 = Conv1D(128, 2, activation=('relu'))(inputs)
x1 = MaxPool1D()(x1)
x2 = MaxPool1D()(x2)
x3 = MaxPool1D()(x3)
x1 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x1)
x2 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x2)
x3 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x3)
x1 = Dense(units=32, activation=('relu'))(x1)
x2 = Dense(units=32, activation=('relu'))(x2)
x3 = Dense(units=32, activation=('relu'))(x3)
x = Concatenate(axis=1)([x1,x2,x3])
x = Dense(units=8, activation=('relu'))(x)
x = Flatten()(x)
x = Dense(units=1, activation=('sigmoid'))(x)
md = Model(inputs, x)
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100)

#method 12: - loss: 0.4679 - accuracy: 0.7858; [[168 100][ 21 294]](0.792)
inputs = Input(shape=(1000,1))
x1 = Conv1D(256, 2, activation=('relu'))(inputs)
x2 = Conv1D(256, 3, activation=('relu'))(inputs)
x3 = Conv1D(128, 2, activation=('relu'))(inputs)
x4 = Conv1D(128, 3, activation=('relu'))(inputs)
x5 = Conv1D(256, 4, activation=('relu'))(inputs)
x6 = Conv1D(128, 4, activation=('relu'))(inputs)
x1 = MaxPool1D()(x1)
x2 = MaxPool1D()(x2)
x3 = MaxPool1D()(x3)
x4 = MaxPool1D()(x4)
x5 = MaxPool1D()(x5)
x6 = MaxPool1D()(x6)
x1 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x1)
x2 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x2)
x3 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x3)
x4 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x4)
x5 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x5)
x6 = BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1)(x6)
x1 = Dense(units=32, activation=('relu'))(x1)
x2 = Dense(units=32, activation=('relu'))(x2)
x3 = Dense(units=32, activation=('relu'))(x3)
x4 = Dense(units=32, activation=('relu'))(x4)
x5 = Dense(units=32, activation=('relu'))(x5)
x6 = Dense(units=32, activation=('relu'))(x6)
x = Concatenate(axis=1)([x1,x2,x3,x4,x5,x6])
x = Dense(units=8, activation=('relu'))(x)
x = Flatten()(x)
x = Dense(units=1, activation=('sigmoid'))(x)
md = Model(inputs, x)
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=10)

#method 13: -loss: 0.4437 - accuracy: 0.8113; [[219  43][ 72 248]](0.8024)
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
x = Dense(units=8, activation=('relu'))(x)
x = Flatten()(x)
x = Dense(units=1, activation=('sigmoid'))(x)
md = Model(inputs, x)
md.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
md.fit(X_train, y_train, batch_size=32, epochs=100)
'''

# Predicting the Test set results
    
