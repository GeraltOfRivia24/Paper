import os,random
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(7)
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import cPickle, random, sys, keras
import keras.backend.tensorflow_backend as KTF
from cnn import *


def get_session(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, visible_device_list = '7')

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
    gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1


def loadData():
    Xd = cPickle.load(open("../rml_data/2016.04C.multisnr.pkl",'rb'))
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = [] 
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(map(lambda data: np.expand_dims(data[0, :], 0), Xd[(mod,snr)]))
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
    X = np.vstack(X)
    print X.shape
    return X, lbl, mods, snrs


def SimpleCNN2(mod, in_shp):
    dr = 0.5 # dropout rate (%)
    model = models.Sequential()
    model.add(Reshape(in_shp+[1], input_shape=in_shp))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(256, [1, 3], border_mode='valid', activation="relu", name="conv1", init='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(80, [1, 3], border_mode="valid", activation="relu", name="conv2", init='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation='relu', init='he_normal', name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense(len(mod), init='he_normal', name="dense2" ))
    model.add(Activation('softmax'))
    model.add(Reshape([len(mod)]))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print model.summary()
    return model


KTF.set_session(get_session())

X, lbl, mods, snrs = loadData()
np.random.seed(2017)
n_examples = X.shape[0]
n_train = int(n_examples * 0.6)
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]

Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

in_shp = list(X_train.shape[1:])

classes = mods

model = SimpleCNN2(mods, in_shp)

filepath = './model/convmodrecnets_CNN2_0.5.wts.h5'
history = model.fit(X_train,
    Y_train,
    batch_size=1024,
    epochs=100,
    verbose=1,
    validation_data=(X_test, Y_test),
    callbacks = [
    keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    ])

# we re-load the best weights once training is finished
model.load_weights(filepath)

acc = {}
for snr in snrs:
    # extract classes @ SNR
    test_SNRs = map(lambda x: lbl[x][1], test_idx)
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    
    
    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print "SNR {} Accuracy {}".format(snr, cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
    
