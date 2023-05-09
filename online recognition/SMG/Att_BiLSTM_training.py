# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy# linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tensorflow import set_random_seed
from keras.models import Model,Sequential
from keras.layers import Input, Dense, Embedding, concatenate
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D
from keras.layers import Add, BatchNormalization, Activation, CuDNNLSTM, Dropout
from keras.layers import *
from keras.models import *
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,Callback
import gc
from sklearn import metrics
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
import matplotlib.pyplot as plt
import scipy.io as sio

numpy.random.seed(1)
set_random_seed(1)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


max_features = 50000

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = inputs.shape[1].value
    SINGLE_ATTENTION_VECTOR = False

    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def attention_spatial_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = inputs.shape[1].value
    SINGLE_ATTENTION_VECTOR = False

    input_dim = int(inputs.shape[2]) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(inputs)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    output_attention_mul = Multiply()([inputs, a])
    return output_attention_mul


def build_model4(timesteps, out_dim,featurelen,units=0, spatial_dr=0.0, dense_units=128, dr=0.1, use_attention=True):

    inp = Input(shape=(timesteps, featurelen))
    x = inp
    #x = attention_spatial_block(x)
    x = Bidirectional(CuDNNGRU(2000, return_sequences=True))(x)
    #x = attention_3d_block(x)
    x = Bidirectional(CuDNNGRU(2000, return_sequences=True))(x)
    #x = attention_spatial_block(x)
    x = Flatten() (x)

    #x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    x = Dense(1000, activation='sigmoid')(x)
    outp = Dense(out_dim, activation="softmax")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])

    return model

def build_model3(timesteps, out_dim,featurelen,units=0, spatial_dr=0.0, dense_units=128, dr=0.1, use_attention=True):
    inp = Input(shape=(timesteps, featurelen))

    x1 = SpatialDropout1D(spatial_dr)(inp)

    x_gru = Bidirectional(CuDNNGRU(units * 2, return_sequences = True))(x1)
    if use_attention:
        x_att = AttLayer(featurelen)(x_gru)
        x = Dropout(dr)(Dense(dense_units, activation='sigmoid') (x_att))
    else:
        x_att = Flatten() (x_gru)
        x = Dropout(dr)(Dense(dense_units, activation='sigmoid') (x_att))

    x = BatchNormalization()(x)
    #x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
    x = Dense(HMM_state_number, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "categorical_crossentropy", optimizer = 'RMSprop', metrics = ["accuracy"])

    return model




def build_model2(timesteps, out_dim,featurelen, units=0, spatial_dr=0.0, dense_units=128, dr=0.1, use_attention=True):
    model = Sequential()
    model.add(Dense(2000,activation='sigmoid',input_shape=(featurelen*timesteps,)))
    model.add(Dropout(0.1))
    model.add(Dense(2000,activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(out_dim ,activation='softmax'))
    model.compile(loss='categorical_crossentropy',
             optimizer='RMSprop',
             metrics=['accuracy'])

    return model

def build_model5(timesteps, out_dim,featurelen, units=0, spatial_dr=0.0, dense_units=128, dr=0.1, use_attention=True):
    model = Sequential()
    model.add(Dense(2000,activation='sigmoid',input_shape=(featurelen*timesteps,)))
    model.add(Dropout(0.1))
    model.add(Dense(2000,activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(out_dim ,activation='softmax'))
    model.compile(loss='categorical_crossentropy',
             optimizer='RMSprop',
             metrics=['accuracy'])

    return model


def build_model1(timesteps, out_dim,featurelen, units=1000, spatial_dr=0.0, dense_units=128, dr=0.1, use_attention=True):

    inp = Input(shape=(timesteps, featurelen))
    x_gru = Bidirectional(CuDNNGRU(units * 2, return_sequences = True))(inp)
    x_gru = attention_3d_block(x_gru)
    x_gru = Bidirectional(CuDNNGRU(units, return_sequences = True))(x_gru)
    x_att = attention_spatial_block(x_gru)
    x = Dense(1000, activation='relu')(x_att)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Flatten() (x)
    outp = Dense(out_dim, activation = "sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def train_att_BiLSTM(parsers, Feature_file):

    batch_size = 128
    epochs = parsers['training_steps']

    #load data
    import pickle
    print ('... loading data')
    f = open(Feature_file,'rb' )
    Feature_train = pickle.load(f)
    f.close()

    # add non gesture state
    dictionaryNum = parsers['class_count']*parsers['STATE_NUM']

    dictionaryNum = dictionaryNum+1

    #READ FEATURES
    Feature_all = Feature_train['Feature_all']

    #read labels
    Target_all = Feature_train['Targets_all']
    #Target_all_numeric = numpy.argmax(Target_all, axis=1)

    timesteps = parsers['LSTM_step']
    featurelen = int(Feature_all.shape[1]/timesteps)
    if parsers['netmodel'] ==  2:
        Feature_all = Feature_all
    else:
        #if using LSTM
        Feature_all = Feature_all.reshape((Feature_all.shape[0],timesteps, featurelen))


    from sklearn.model_selection import train_test_split
    X_tra, X_val, y_tra, y_val = train_test_split(Feature_all, Target_all, test_size = 0.1, random_state=42)


    ModelCheckpointpath = parsers['outpath']+ parsers['experi_ID'] +'/model4.model'
    early_stopping = EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='max')
    model_checkpoint = ModelCheckpoint(ModelCheckpointpath, save_best_only=True, verbose=1, monitor='acc', mode='max')
    reduce_lr = ReduceLROnPlateau(factor=0.5, patience=1, min_lr=0.0001, verbose=1)
    his = LossHistory()
    callbacks = [early_stopping, model_checkpoint,reduce_lr,his]


    #MODEL4 = build_model3()
    if parsers['netmodel'] == 1:
        model = build_model1(timesteps= timesteps, out_dim = dictionaryNum,featurelen=featurelen , spatial_dr = 0.1, dense_units=1000, dr=0.1, use_attention=True)
    if parsers['netmodel'] == 2:
        model = build_model2(timesteps= timesteps, out_dim = dictionaryNum,featurelen=featurelen , spatial_dr = 0.1, dense_units=1000, dr=0.1, use_attention=True)
    if parsers['netmodel'] == 4:
        model = build_model4(timesteps= timesteps, out_dim = dictionaryNum,featurelen=featurelen , spatial_dr = 0.1, dense_units=1000, dr=0.1, use_attention=True)

    model.summary()
    model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), verbose=True,callbacks=callbacks)

    Modelpath = parsers['outpath']+ parsers['experi_ID'] +'/my_model.h5'
    model.save(filepath = Modelpath )


    #print(his.losses)

    # summarize history for accuracy
#    plt.plot(his.losses)
#    plt.plot(his.acc)
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('batch')
#    plt.legend(['train', 'test'], loc='upper left')
#    plt.show()
#
    #del MODEL4  # deletes the existing model

    modelname = Modelpath

    del model  # deletes the existing model

#    pred_val_y = model.predict([X_val], batch_size=1024, verbose=1)
    best_threshold = 0# = evaluate(y_val, pred_val_y)

    return modelname,best_threshold

def evaluate(y, pred):
    f1_list = list()
    thre_list = numpy.arange(0.1, 0.501, 0.01)
    for thresh in thre_list:
        thresh = numpy.round(thresh, 2)
        f1 = metrics.f1_score (y_true = y, y_pred = (pred>thresh).astype(int), labels=y, pos_label=1, average=None,sample_weight=None)
        f1_list.append(f1)
        print("F1 score at threshold {0} is {1}".format(thresh, f1))
    #return f1_list
    plot_confusion_matrix(y, numpy.array(pd.Series(pred.reshape(-1,)).map(lambda x:1 if x>thre_list[numpy.argmax(f1_list)] else 0)))
    print('Best Threshold: ',thre_list[numpy.argmax(f1_list)])
    return thre_list[numpy.argmax(f1_list)]

def test_att_BiLSTM(MODEL4, parsers, Feature_all,best_threshold):

    #Target_all_numeric = numpy.argmax(Target_all, axis=1)
    timesteps = parsers['LSTM_step']

    featurelen = parsers['featureNum']

    if parsers['netmodel'] ==  2:
        Feature_all = Feature_all
    else:
        #if using LSTM
        Feature_all = Feature_all.reshape((Feature_all.shape[0], timesteps, featurelen))

    y_pred_4 = MODEL4.predict(Feature_all, batch_size=2000, verbose=0)

    #y_pred_4 = (y_pred_4 > 0.38) * y_pred_4
    del MODEL4  # deletes the existing model

    return y_pred_4
