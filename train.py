# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 20:35:31 2020
dnn1中model.compile所用的metric考虑其他的，而不是accuracy，根据先验的不同类别分布设定合适的metric
model = Sequential()
model.add(Dense(1024,input_dim=41,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(256,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(64,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(16,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(4,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(1))
model.add(Activation('sigmoid'))
@author: CP3
"""

from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

traindata = pd.read_csv('kddtrain.csv', header=None)
testdata = pd.read_csv('kddtest.csv', header=None)


X = traindata.iloc[:,np.r_[1:20,22:42]]
Y = traindata.iloc[:,0]
#C = testdata.iloc[:,0]
#T = testdata.iloc[:,1:42]

trainX = np.array(X)
#testT = np.array(T)

trainX.astype(float)
#testT.astype(float)

#scaler = Normalizer().fit(trainX)
#trainX = scaler.transform(trainX)

#scaler = Normalizer().fit(testT)
#testT = scaler.transform(testT)

y_train = np.array(Y)
#y_test = np.array(C)

#take log 0,4,5          12,15,20,21,29,30
for row in [0,4,5]:
    trainX[:,row] = np.log10(trainX[:,row]+1)

minX = np.min(trainX,0)
maxX = np.max(trainX,0)

for row in range(len(minX)):
    trainX[:,row] = (trainX[:,row]-minX[row])/(maxX[row]-minX[row])
    
X_train = np.array(trainX)
#X_test = np.array(testT)


Batch_size = 64
dropout = 0.01
# 1. define the network
model = Sequential()
model.add(Dense(1024,input_dim=39,activation='relu'))  
model.add(Dropout(dropout))
#model.add(Dense(768,activation='relu'))  
#model.add(Dropout(dropout))
#model.add(Dense(512,activation='relu'))  
#model.add(Dropout(dropout))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="test/preprocess/dnn1layer_64_dr_0.01/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('test/preprocess/dnn1layer_64_dr_0.01_csv/training_set_dnnanalysis.csv',separator=',', append=False)
model.fit(X_train, y_train, batch_size=Batch_size, epochs=100, callbacks=[checkpointer,csv_logger])
model.save("test/preprocess/dnn1layer_64_dr_0.01/dnn3layer_model.hdf5")


"""
accuracy
0.933
precision
0.998
recall
0.918
f1score
0.956
"""















"""
import numpy as np
import pandas as pd
#from sklearn.kernel_approximation import RBFSampler
#from sklearn.linear_model import SGDClassifier
#from sklearn.cross_validation import train_test_split
from sklearn import svm
#from sklearn.metrics import classification_report
#from sklearn import metrics
from sklearn.linear_model import LogisticRegression
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
#from sklearn.model_selection import GridSearchCV
#from sklearn.svm import SVC
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error, roc_curve, classification_report,auc)

traindata = pd.read_csv('kddtrain.csv', header=None)
testdata = pd.read_csv('kddtest.csv', header=None)

X = traindata.iloc[:,1:42] #41 features
Y = traindata.iloc[:,0] #1 attack 0 normal
C = testdata.iloc[:,0]
T = testdata.iloc[:,1:42]

scaler = Normalizer().fit(X) #do nothing
trainX = scaler.transform(X) #normalize with l2 norm

scaler = Normalizer().fit(T) #do nothing
testT = scaler.transform(T) #normalize with l2 norm


traindata = np.array(trainX)
trainlabel = np.array(Y)

testdata = np.array(testT)
testlabel = np.array(C)



#traindata = X_train
#testdata = X_test
#trainlabel = y_train
#testlabel = y_test






model = svm.SVC(kernel='rbf',probability=True)
model = model.fit(traindata, trainlabel)

# make predictions
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)
#np.savetxt('classical/predictedlabelSVM-rbf.txt', predicted, fmt='%01d')
#np.savetxt('classical/predictedprobaSVM-rbf.txt', proba)

print("--------------------------------------SVMrbf--------------------------------------")
y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

"""
#对于类别不均衡的分类模型，采用macro方式会有较大的偏差，采用weighted方式则可较好反映模型的优劣，因为若类别数量较小则存在蒙对或蒙错的概率，其结果不能真实反映模型优劣，需要较大的样本数量才可计算较为准确的评价值，通过将样本数量作为权重，可理解为评价值的置信度，数量越多，其评价值越可信。
"""

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)
model = svm.SVC(kernel='linear', C=1000,probability=True)
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
proba = model.predict_proba(testdata)

#np.savetxt('classical/predictedlabelSVM-linear.txt', predicted, fmt='%01d')
#np.savetxt('classical/predictedprobaSVM-linear.txt', proba)

# summarize the fit of the model
print("--------------------------------------SVM linear--------------------------------------")
y_train1 = expected
y_pred = predicted
accuracy = accuracy_score(y_train1, y_pred)
recall = recall_score(y_train1, y_pred , average="binary")
precision = precision_score(y_train1, y_pred , average="binary")
f1 = f1_score(y_train1, y_pred, average="binary")

print("accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f1score")
print("%.3f" %f1)
#
#print("-----------------------------------------LR---------------------------------")
#model = LogisticRegression()
#model.fit(traindata, trainlabel)
#
## make predictions
#expected = testlabel
#np.savetxt('classical/expected.txt', expected, fmt='%01d')
#predicted = model.predict(testdata)
#proba = model.predict_proba(testdata)
#
#np.savetxt('classical/predictedlabelLR.txt', predicted, fmt='%01d')
#np.savetxt('classical/predictedprobaLR.txt', proba)
#
#y_train1 = expected
#y_pred = predicted
#accuracy = accuracy_score(y_train1, y_pred)
#recall = recall_score(y_train1, y_pred , average="binary")
#precision = precision_score(y_train1, y_pred , average="binary")
#f1 = f1_score(y_train1, y_pred, average="binary")
#
#print("accuracy")
#print("%.3f" %accuracy)
#print("precision")
#print("%.3f" %precision)
#print("recall")
#print("%.3f" %recall)
#print("f1score")
#print("%.3f" %f1)
#
"""