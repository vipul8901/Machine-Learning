# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 07:09:46 2020

@author: Vipul
"""

import sys


import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import os
if os.path.exists("output3.csv"):
  os.remove("output3.csv")

inp = pd.read_csv("input3.csv")

color={1:"red",0:"blue"}
plt.scatter(inp.A,inp.B,c=inp.label.map(color))

#Train test splitS


x=inp[["A","B"]]
y=inp[["label"]]

x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,train_size=.6)


#model build


"""clf=svm.SVC(kernel='linear',C=100) #accuracy= .59 ie all classified as 1 class
#clf=svm.SVC(kernel='rbf',C=1, gamma=1) #accuracy= .96
#clf=svm.SVC(kernel='poly',C=1, gamma=1) #accuracy= .59
#clf.fit(x_train,y_train)
clf.fit(x_train,y_train.label)

pred=clf.predict(x_test)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, pred))"""




def estimator(name,model,parameters):
    global clf
    clf=GridSearchCV(model, parameters,cv=5)
    clf.fit(x_train,y_train.label)
    
    pred0=clf.predict(x_train)
    print("Accuracy Train:",metrics.accuracy_score(y_train, pred0))
    trn=metrics.accuracy_score(y_train, pred0)
    
    pred=clf.predict(x_test)
    print("Accuracy Test:",metrics.accuracy_score(y_test, pred))
    tst=metrics.accuracy_score(y_test, pred)
    
    with open("output3.csv", 'a', newline='') as csvfile:
            fieldnames = ['name', 'train', 'test']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'name': name, 'train': trn, 'test': tst})


svc=svm.SVC()
param = {'kernel':['linear'], 'C':[0.1, 0.5, 1, 5, 10, 50, 100]}
estimator("svm_linear",svc,param)
"""
Accuracy Train: 0.59
Accuracy Test: 0.59
"""


param= {'kernel':['poly'], 'C':[0.1, 1, 3], 'degree':[4,5,6], 'gamma':[0.1, 0.5]} #.67
estimator("svm_polynomial",svc,param)
"""
Accuracy Train: 0.7566666666666667
Accuracy Test: 0.675
"""



param = {'kernel':['rbf'], 'C':[0.1, 0.5, 1, 5, 10, 50, 100], 'gamma':[0.1, 0.5, 1, 3, 6, 10]} #.985
estimator("svm_rbf",svc,param)
"""
Accuracy Train: 0.9766666666666667
Accuracy Test: 0.985
"""

modl=LogisticRegression()
param = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}
estimator("logistic",modl,param)
"""
Accuracy Train: 0.59
Accuracy Test: 0.59
"""

modl=KNeighborsClassifier()
param = {'n_neighbors':list(range(1,51)), 'leaf_size':list(range(5,65,5))}
estimator("knn",modl,param)
"""
Accuracy Train: 1.0
Accuracy Test: 0.945
"""

modl=DecisionTreeClassifier()
param = {'max_depth':list(range(1,51)), 'min_samples_split':list(range(2,11))}
estimator("decision_tree",modl,param)
"""
Accuracy Train: 1.0
Accuracy Test: 0.995
"""


modl=RandomForestClassifier()
param = {'max_depth':list(range(1,51)), 'min_samples_split':list(range(2,11))}
estimator("random_forest",modl,param)
"""
Accuracy Train: 1.0
Accuracy Test: 0.98
"""



"""plt.subplot(1, 1, 1)
plt.contourf(x_test.A, x_test.B, pred, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(x_test.A, x_test.B, c=y_test.label, cmap=plt.cm.Paired)
plt.show()"""

#plt.scatter(x_train.A, x_train.B, c=y_train.label, s=30, cmap=plt.cm.Paired)