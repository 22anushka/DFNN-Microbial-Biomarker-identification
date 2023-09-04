```
@author Anushka S
```

# Code reference: https://github.com/MicroAVA/GEDFN/blob/master/keras_gedfn.py

import keras
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import shuffle
from keras.layers import Dense, Input, Dropout, BatchNormalization, Activation, LeakyReLU
from keras.models import Model
import tensorflow.keras.utils as ku
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.regularizers import l1, l2, l1_l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import tensorflow as tf
from keras.callbacks import Callback
from Sparse import Sparselayer
import random


# source: https://stackoverflow.com/questions/37293642/how-to-tell-keras-stop-training-based-on-loss-value
class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.1, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
			

def gedfn(x_train, x_test, y_train, y_test, sparse_connection, nameOf):

    max_norm_constraint = keras.constraints.max_norm(3.)
    # x_train, y_train = utils.upsampling(x_train, y_train)
    input = Input(shape=(np.shape(x_train)[1],), name='input')
    # in_put = Dropout(0.1)(input)
    L1 = BatchNormalization()(input)

    L1 = Sparselayer(adjacency_mat=sparse_connection,kernelInitializer='he_uniform', kernelConstraint=1, nameOfLayer='L1', inp=L1)
    # kernel_constraint=MyConstraint(my_constraints)
    # L1 = BatchNormalization()(L1) # don't use!
    L1 = Activation('relu')(L1)

    L2 = Dense(128,kernel_initializer='he_uniform',name='L2')(L1)
    L2 = BatchNormalization()(L2)
    L2 = Activation('relu')(L2)
    # L2 = LeakyReLU(alpha=0.3)(L2)
    L2 = Dropout(0.5)(L2)

    L3 = Dense(32,kernel_initializer='he_uniform',name='L3')(L2)
    L3 = BatchNormalization()(L3)
    L3 = Activation('relu')(L3)
    # L3 = LeakyReLU(alpha=0.3)(L3)
    L3 = Dropout(0.5)(L3)

    L4 = Dense(8,kernel_initializer='he_uniform',name='L4')(L3)
    L4 = BatchNormalization()(L4)
    L4 = Activation('relu')(L4)
    # L4 = LeakyReLU(alpha=0.3)(L4)
    L4 = Dropout(0.5)(L4)


    output = Dense(1, activation='sigmoid', name='output')(L4)

    model = Model(inputs=[input], outputs=[output])

    ku.plot_model(model, to_file='gemlp_model.png', show_shapes=True)
    # ada = keras.optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
    adam = tf.keras.optimizers.Adam(lr=0.0001)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    # earlyStopping = early_stop.LossCallBack(loss=0.1)
    callbacks = [
    EarlyStoppingByLossVal(monitor='val_loss', value=0.1, verbose=1),
    # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    # ModelCheckpoint('/content/checkpoint', monitor='val_loss', save_best_only=True, verbose=0),
    ]

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                        epochs=200, verbose=1,callbacks=callbacks,
                        batch_size=32)


    test_loss, test_acc = model.evaluate(x_test, y_test)
    #
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()
    plt.show()

    y_predict = model.predict(x_test)

    left_embedding_layer_weights = model.layers[2].get_weights()[0]
    right_embedding_layer_weights = model.layers[4].get_weights()[0]

    gamma_c = 50
    gamma_numerator = np.sum(sparse_connection, axis=0)
    print(gamma_numerator)
    gamma_denominator = np.sum(sparse_connection, axis=0)
    filtered = np.where(gamma_numerator > gamma_c)
    for i in filtered:
      gamma_numerator[i] = gamma_c 

    var_left = tf.reduce_sum(tf.abs(tf.multiply(left_embedding_layer_weights, sparse_connection)), 0)
    var_right = tf.reduce_sum(tf.abs(right_embedding_layer_weights), 1)
    var_importance = tf.add(tf.multiply(tf.multiply(var_left, gamma_numerator), 1. / gamma_denominator),
                            tf.multiply(tf.multiply(var_right, gamma_numerator), 1. / gamma_denominator))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        var_imp = sess.run([var_importance])
        var_imp = np.reshape(var_imp, [np.shape(x_train)[1]])
        
        np.savetxt("/content/l1_weights"+nameOf+".txt", left_embedding_layer_weights, delimiter=",")
        np.savetxt("/content/l2_weights"+nameOf+".txt", right_embedding_layer_weights, delimiter=",")
        np.savetxt("/content/var_results"+nameOf+".csv", var_imp, delimiter=",")

    #model.save("/content/model"+nameOf)
    return y_predict, test_loss, test_acc


dataset = pd.read_csv('/content/otu_IBD.csv', index_col=0, header=0)
# below lines (140-151) only for IBD dataset, for CRC dataset, let dataset_new = dataset
# remove columns from the file that dont match:
# column number as a result from MAGMA network construction
# x_test = X.sample(frac=1).reset_index(drop=True)
print(dataset)
with open('/content/col_list_10%.csv') as f:
    col_list = f.read().splitlines()
label_list = []
# print(col_list)
print(dataset.columns)
[label_list.append(dataset.columns[int(i)-1]) for i in col_list]
dataset_new = dataset.drop(columns=label_list)
print(dataset_new)


# split the dataset
train, test = train_test_split(dataset_new, test_size=0.2)
y_train = train['label'].values
x_train = train.drop(columns=['label'])

y_test = test['label'].values
x_test = test.drop(columns=['label'])


print(x_test.shape)
print(x_train.shape)

method = pd.read_csv('/content/MAGMA_new.csv',index_col=0, header=0)

resultsA = gedfn(x_train, x_test, y_train, y_test, method, "magma")
print("Accuracy: ",resultsA[2])
print("Loss: ", resultsA[1])
print("ROC-AUC: ", roc_auc_score(y_test,resultsA[0]))



