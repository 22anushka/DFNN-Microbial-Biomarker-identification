
!pip install deep-forest

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 07:42:02 2022

@author: syama
"""
from pickle import load
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time

#classification
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as acc

#MLP

import keras
import keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from keras.models import Model, load_model

import DNN_models

#DF and XGB
import xgboost as xgb
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,confusion_matrix,roc_curve
import time
from copy import deepcopy
import sys

from xgboost import XGBClassifier
# from deepforest import CascadeForestClassifier
import deepforest


roc = [0,0,0,0,0]
accuracy = [0,0,0,0,0]
precision_val = [0,0,0,0,0]
recall_val = [0,0,0,0,0]
f1_val = [0,0,0,0,0]

def get_dataset():
    #otu_data_file = '/content/otu_IBD.csv'
    otu_data_file = '/content/otu_table_formatted_CRC.csv'
    
    #otu_data_file = '/content/otu_table_formatted_autism5%.csv'

    dataFile = pd.read_csv(otu_data_file, sep=',')
    #print(dataFile)
    print(dataFile.shape)
    
    Y = dataFile['label']
    Y=np.array(Y)
    Y=Y.reshape(Y.shape[0],-1)
    '''
    # for IBD Dataset
    # # otu_data_file='Dataset/'+disease+'.txt'
    df = dataFile.drop(['label'],axis=1)
    df = df.drop(['SampleID'],axis=1)
    with open('/content/col_list_10%.csv') as f:
      col_list = f.read().splitlines()
    label_list = []
    # print(col_list)
    # print(dataset.columns)
    [label_list.append(df.columns[int(i)-1]) for i in col_list]
    df = df.drop(columns=label_list)
    '''
    # for other dataset
    df = dataFile.drop(['label'],axis=1)
    df = df.drop(['Sample ID'],axis=1)
    
    X=df

    print(df.shape)
    
    # # ##to get column names
    # # df.columns = df.iloc[0]#all column names will be here
    # old_features=df.columns
    # print(old_features)
    

    # # X = df.iloc[1: , :]

    
    # # print(f'data size{X.shape}')
    # # print(f'Labelfile size{Y.shape}')
    
    return X,Y
def classification( X_train, X_test, y_train, y_test,n,hyper_parameters, j, method='svm', cv=5, scoring='roc_auc', n_jobs=1, cache_size=10000):
        j = int(j)
        #clf_start_time = time.time()
        disease = "Biomarker_crc"
        k=n
        feat_algo = "GDEFN"
        print("# Tuning hyper-parameters")
        print(X_train.shape, y_train.shape)

        
        # Support Vector Machine
        if method == 'svm':
            #clf = GridSearchCV(SVC(probability=True, cache_size=cache_size), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100, )
            #clf.fit(X_train, y_train)
            clf = SVC(probability=True)
            clf.fit(X_train,y_train)
        
        # Random Forest
        if method == 'rf':
            clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, random_state=0), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100)
            clf.fit(X_train, y_train)
        
        # Multi-layer Perceptron
        if method == 'mlp':
            model = KerasClassifier(build_fn=DNN_models.mlp_model, input_dim=X_train.shape[1], verbose=0, )
            clf = GridSearchCV(estimator=model, param_grid=hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100)
            clf.fit(X_train, y_train, batch_size=32)
            
        
        print("Best parameters set found on development set:")
        #print(clf.best_params_)

        # Evaluate performance of the best model on test set
        y_true, y_pred = y_test, clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
        '''metrics= [round(roc_auc_score(y_true, y_prob[:, 1]), 4),
                   round(accuracy_score(y_true, y_pred), 4),
                   round(precision_score(y_true, y_pred), 4),
                   round(recall_score(y_true, y_pred), 4),
                   round(f1_score(y_true, y_pred), 4),]'''

        roc[j] += round(roc_auc_score(y_true, y_prob[:, 1]), 4)
        accuracy[j] += round(accuracy_score(y_true, y_pred), 4)
        precision_val[j] += round(precision_score(y_true, y_pred), 4)
        recall_val[j] += round(recall_score(y_true, y_pred), 4)
        f1_val[j] += round(f1_score(y_true, y_pred), 4)

        # time stamp
       # metrics.append(str(datetime.datetime.now()))

        # running time
        #metrics.append(round( (time.time() - self.t_start), 2))

        # classification time
        #metrics.append(round( (time.time() - clf_start_time), 2))

        # best hyper-parameter append
        #metrics.append(str(clf.best_params_))

        # Write performance metrics as a file
        '''res = pd.DataFrame([metrics], index=[disease+"_" + feat_algo +"_"+str(k)+ "_"+ method])
        with open("/content/results_3" + disease + "-Magma_AllSamples.txt", 'a') as f:
            res.to_csv(f, header=None)
            

        print('Accuracy metrics')
        print('AUC, ACC, Precision, Recall, F1_score, time-end, runtime(sec), classfication time(sec), best hyper-parameter')
        print(metrics)
        return metrics'''
        
def deep_forest(X_train,y_train,X_test,y_test,f, j):
    j = int(j)
    model = deepforest.CascadeForestClassifier(random_state=1)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    roc[j] += roc_auc_score(y_test,y_prob[:,1])
    accuracy[j] += accuracy_score(y_test, y_pred)
    precision_val[j] += precision_score(y_test, y_pred)
    recall_val[j] += recall_score(y_test, y_pred)
    f1_val[j] += f1_score(y_test, y_pred)
    '''metrics_gc=str(auc) + ',' + str(acc) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' + 'Deep'
    f.write(metrics_gc)
    f.write('\n')'''
    
def xg_boost(X_train,y_train,X_test,y_test,f, j):
    j = int(j)
    #XGBOOST2

    #dictionary for collecting results
    results_dict = {}
    
    #obtaining default parameters by calling .fit() to XGBoost model instance
    xgbc0 = xgb.XGBClassifier(objective='binary:logistic',
                              booster='gbtree',
                              eval_metric='auc',
                              tree_method='hist',
                              grow_policy='lossguide',
                              use_label_encoder=False)
    xgbc0.fit(X_train , y_train)
    
    #extracting default parameters from benchmark model
    default_params = {}
    gparams = xgbc0.get_params()
    
    #default parameters have to be wrapped in lists - even single values - so GridSearchCV can take them as inputs
    for key in gparams.keys():
        gp = gparams[key]
        default_params[key] = [gp]
    
    #benchmark model. Grid search is not performed, since only single values are provided as parameter grid.
    #However, cross-validation is still executed
    clf0 = GridSearchCV(estimator=xgbc0, scoring='accuracy', param_grid=default_params, return_train_score=True, verbose=1, cv=5)
    clf0.fit(X_train, y_train.ravel())
    
    #results dataframe
    df = pd.DataFrame(clf0.cv_results_)
    
    #predictions - inputs to confusion matrix
    train_predictions = clf0.predict(X_train)
    test_predictions = clf0.predict(X_test)
    #unseen_predictions = clf0.predict(df_test.iloc[:,1:])
    
    #confusion matrices
    cfm_train = confusion_matrix(y_train, train_predictions)
    cfm_test = confusion_matrix(y_test, test_predictions)
    #cfm_unseen = confusion_matrix(df_test.iloc[:,:1], unseen_predictions)
    
    #accuracy scores
    accs_train = accuracy_score(y_train, train_predictions)
    accs_test = accuracy_score(y_test, test_predictions)
    accuracy[j] += accs_test
    #accs_unseen = accuracy_score(df_test.iloc[:,:1], unseen_predictions)
    
    #F1 scores for each train/test label
    f1s_train_p1 = f1_score(y_train, train_predictions, pos_label=1)
    f1s_train_p0 = f1_score(y_train, train_predictions, pos_label=0)
    f1s_test_p1 = f1_score(y_test, test_predictions)
    f1_val[j] += f1s_test_p1
    recall_val[j] += recall_score(y_test, test_predictions)
    precision_val[j] += precision_score(y_test, test_predictions)
         
    test_ras = roc_auc_score(y_test, clf0.predict_proba(X_test)[:,1])
    roc[j] += test_ras

    bp = clf0.best_params_
    
    results_dict['xgbc0'] = {'iterable_parameter': np.nan,
                             'classifier': deepcopy(clf0),
                             'cv_results': df.copy(),
                             'cfm_train': cfm_train,
                             'cfm_test': cfm_test,
                             # 'cfm_unseen': cfm_unseen,
                             # 'train_accuracy': accs_train,
                             'test_accuracy': accs_test,
                             # 'unseen_accuracy': accs_unseen,
                             # 'train F1-score label 1': f1s_train_p1,
                             # 'train F1-score label 0': f1s_train_p0,
                             'test F1-score label 1': f1s_test_p1,
                             # 'test F1-score label 0': f1s_test_p0,
                             # 'unseen F1-score label 1': f1s_unseen_p1,
                             # 'unseen F1-score label 0': f1s_unseen_p0,
                             'test roc auc score': test_ras,
                             'Recall':recall_val,
                             'Precision': precision_val,
                             # 'unseen roc auc score': unseen_ras,
                             'best_params': bp}
    print("finished")
    #print(results_dict)
    '''
    metrics=str(test_ras) + ',' + str(accs_test) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1s_test_p1)
    f.write(metrics)
    f.write('\n') '''       
	
	
X,y=get_dataset()
y=y.ravel()
print(y.shape)
print("Hi")
svm_hyper_parameters = [
                        {'C': [2 ** s for s in range(-5, 16, 2)], 'gamma': [2 ** s for s in range(3, -15, -2)],
                        'kernel': ['rbf']}] 
rf_hyper_parameters =   [{'n_estimators': [100], 'max_features': ['log2'],'min_samples_leaf': [1, 2],
                         'criterion': ['gini'] }]   
mlp_hyper_parameters = [{'numHiddenLayers': [ 3],
                            'epochs': [ 50],
                            'numUnits': [ 30],
                            'dropout_rate': [0.1],
                            },]    

col_filename = ''

#with open('/content/var_ibdsparcc_withcols.csv') as f:

with open(col_filename) as f:
  features = f.read().splitlines()
for iter in range(5):  
    # comment out for loop for all samples, only run 317-322 with j=0. All results in first array item.
    for i in range(100, 501, 100):  
      print("Top {} features".format(i))
      temp = (X.columns).tolist()
      for k in features[0:i]:
        temp.remove(k)
      X_new = X.drop(columns=temp)
      print(X_new.shape)
    
      X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2) # split dataset
      # fit
      
      # run one at a time
      #metrics=classification(X_train, X_test, y_train, y_test,i, hyper_parameters=rf_hyper_parameters, j = (i/100)-1, method='rf', cv=5,n_jobs=-1, scoring='roc_auc')
      #metrics=classification(X_train, X_test, y_train, y_test,i, hyper_parameters=mlp_hyper_parameters,j = (i/100)-1, method='mlp', cv=5,n_jobs=-1, scoring='accuracy')
      #metrics=classification(X_train, X_test, y_train, y_test,i,hyper_parameters=svm_hyper_parameters, j = (i/100)-1,method='svm', cv=5, n_jobs=-1, scoring='roc_auc', cache_size=1000)
      #deep_forest(X_train,y_train,X_test,y_test, f, i/100 -1)   
      #xg_boost(X_train,y_train,X_test,y_test, f, i/100 -1)
      print("classification done\n")
	  
print([item / 5 for item in roc], [item / 5 for item in accuracy], [item / 5 for item in precision_val], [item / 5 for item in recall_val], [item / 5 for item in f1_val])
        
