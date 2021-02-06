import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wave

import librosa
import librosa.display

import theano
import keras
from keras.layers import Dense, SimpleRNN, BatchNormalization, Input, Dropout, LSTM,TimeDistributed
from keras.optimizers import Adam, Adagrad, Adadelta, RMSprop
from keras import Model
from keras.metrics import Precision, Recall

train_pos_dir = '/Users/shryansgoyal/Downloads/CoughNetData/train/pos/'
train_neg_dir = '/Users/shryansgoyal/Downloads/CoughNetData/train/neg/'

test_pos_dir = '/Users/shryansgoyal/Downloads/CoughNetData/test/pos/'
test_neg_dir = '/Users/shryansgoyal/Downloads/CoughNetData/test/neg/'

val_pos_dir = '/Users/shryansgoyal/Downloads/CoughNetData/val/pos/'
val_neg_dir = '/Users/shryansgoyal/Downloads/CoughNetData/val/neg/'


def get_melspectrogram_db(file_path, sr=16000, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    wav,sr = librosa.load(file_path,sr=sr)
    if wav.shape[0]<5*sr:
        wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
    else:
        wav=wav[:5*sr]
    spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db

def create_melspect(dir_name, label):
    matrix = []
    counter = 0
    for file in os.listdir(dir_name):
         filename = os.fsdecode(file)
         if filename.endswith(".wav"):
            mel_spect = get_melspectrogram_db(dir_name+filename)
            result = np.zeros((128, 256))
            result[:mel_spect.shape[0],:mel_spect.shape[1]] = mel_spect
            matrix.append(result)
            counter += 1
    label = np.array([label]*counter).reshape(counter, -1)
    final = np.array(matrix)
    
    p = np.random.permutation(counter)
    
    return final[p],label[p]

X_train_pos, y_train_pos = create_melspect(train_pos_dir, 1)
X_train_neg, y_train_neg = create_melspect(train_neg_dir, 0)
X_train = np.vstack((X_train_pos,X_train_neg))
y_train = np.vstack((y_train_pos,y_train_neg))

X_val_pos, y_val_pos = create_melspect(val_pos_dir, 1)
X_val_neg, y_val_neg = create_melspect(val_neg_dir, 0)
X_val = np.vstack((X_val_pos,X_val_neg))
y_val = np.vstack((y_val_pos,y_val_neg))

X_test_pos, y_test_pos = create_melspect(test_pos_dir, 1)
X_test_neg, y_test_neg = create_melspect(test_neg_dir, 0)
X_test = np.vstack((X_test_pos,X_test_neg))
y_test = np.vstack((y_test_pos,y_test_neg))

print(f'X_train dim: {X_train.shape}, y_train dim: {y_train.shape}')
print(f'X_val dim: {X_val.shape}, y_val dim: {y_val.shape}')
print(f'X_test dim: {X_test.shape}, y_test dim: {y_test.shape}')


theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

print(f'Beginning training model now.....')

adam = Adam(lr=0.1)
i = Input(shape=(128,256))
d1 = Dense(64, activation="sigmoid")(i)
b1 = BatchNormalization(momentum=0.99, epsilon=0.001)(d1)
# d2 = Dense(64)(d1)
rnn = SimpleRNN(64)(b1) ## Replace this with LSTM(64) if need to run LSTM
d3 = Dense(32, activation="sigmoid")(rnn)
d4 = Dense(1, activation="softmax")(d3)

model = Model(inputs=i, outputs=d4)

print(model.count_params())
print(model.summary())

model.compile(metrics=['acc'],loss='binary_crossentropy', optimizer=adam)
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))

