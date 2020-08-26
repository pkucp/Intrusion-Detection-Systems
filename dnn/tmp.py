# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:13:35 2020

@author: Administrator
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

traindata = pd.read_csv('kdd/binary/Training.csv', header=None)
testdata = pd.read_csv('kdd/binary/Testing.csv', header=None)


X = traindata.iloc[:,np.r_[1:20,22:42]]
Y = traindata.iloc[:,0]
C = testdata.iloc[:,0]
T = testdata.iloc[:,np.r_[1:20,22:42]]

trainX = np.array(X)
testT = np.array(T)

trainX.astype(float)
testT.astype(float)
minX = np.min(trainX,0)
maxX = np.max(trainX,0)
for row in range(len(minX)):
    trainX[:,row] = (trainX[:,row]-minX[row])/(maxX[row]-minX[row])