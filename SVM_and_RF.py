# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:14:52 2019

@author: LENOVO
"""

import matplotlib.pyplot as plt
import numpy as np
## Import sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.model_selection import GridSearchCV


x_train = np.loadtxt("result/V3_dropout0.5/x_train2.txt")
y_train = np.loadtxt("result/V3_dropout0.5/y_train2.txt")
x_test = np.loadtxt("result/V3_dropout0.5/x_test2.txt")
y_test = np.loadtxt("result/V3_dropout0.5/y_test2.txt")



## For V1 dropout0.5, SVM best {'C': 25, 'gamma': 0.0001, 'kernel': 'sigmoid'}
## For V2 dropout0.5, SVM best {'gamma': 0.001, 'C': 10, 'kernel': 'rbf'}
## For V3 dropout0.5, SVM best {'gamma': 0.0001, 'C': 1000, 'kernel': 'rbf'}


## Let user to enter the best hype-parameter
Kernel_best = input("Please enter the best kernel: ")
C_best = float(input("Please enter the best C: "))
gamma_best = float(input("Please enter the best gamma: "))


# Use SVM
def SVM_Sk(x_train, x_test, y_train, y_test):   
    # Define model
    clf = svm.SVC(C=C_best, kernel=Kernel_best, gamma=gamma_best, random_state=123)
    # Training model
    clf.fit(x_train, y_train)
    # Predict
    y_pred = clf.predict(x_test)
    # Calculate accuracy
    
    false_positive_rate, true_positive_rate, thresholds_noscale = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    print('SVM Accuracy = {:0.2f}%, F1_score = {:0.2f}, AUC = {:0.2f}'.format(
            100 * metrics.accuracy_score(y_test, y_pred), 
            f1_score(y_test, y_pred),
            roc_auc))    

    plt.figure(figsize = (6,6))
    plt.title('Receiver Operating Characteristic')
    #plt.plot(false_positive_rate_noscale, true_positive_rate_noscale, 'b',label='Non-scaled features AUC = %0.2f'% roc_auc_noscale,color="blue")
    plt.plot(false_positive_rate, true_positive_rate, 'b',label='Scaled features AUC = %0.2f'% roc_auc,color="green")
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--',label='Random Guessing Line',color="red")
    plt.legend(loc='lower right')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.show()

SVM_Sk(x_train, x_test, y_train, y_test)






def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    # Calculate accuracy    
    false_positive_rate, true_positive_rate, thresholds_noscale = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    print('Random Forest Accuracy = {:0.2f}%, F1_score = {:0.2f}, AUC = {:0.2f}'.format(
            100 * metrics.accuracy_score(y_test, y_pred), 
            f1_score(y_test, y_pred),
            roc_auc))

    plt.figure(figsize = (6,6))
    plt.title('Receiver Operating Characteristic')
    #plt.plot(false_positive_rate_noscale, true_positive_rate_noscale, 'b',label='Non-scaled features AUC = %0.2f'% roc_auc_noscale,color="blue")
    plt.plot(false_positive_rate, true_positive_rate, 'b',label='Scaled features AUC = %0.2f'% roc_auc,color="green")
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--',label='Random Guessing Line',color="red")
    plt.legend(loc='lower right')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.show()
    

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(x_train, y_train)
best_grid = grid_search.best_estimator_
evaluate(best_grid, x_test, y_test)



