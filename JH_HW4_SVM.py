#!/usr/bin/env python3
# Jingyi hui 11/24/2018
# CSCI573 DATAMINING
# HOMEWORK 4 SVM

import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
from statistics import mean, stdev


###################################################
# load the data file
df = pd.read_csv('house-votes-84.data', header=None)
row, column = df.shape
# print(df.head())
df.replace('n', -1, inplace=True)
df.replace('y', 1, inplace=True)
df.replace('?', 0, inplace=True)

###################################################
# split dataset into 1:3 for tuning and 3—fold respectively
X = np.array(df.loc[:, 1:])
y = np.array(df.loc[:, 0])
X_tune, X_3_fold, y_tune, y_3_fold = train_test_split(X, y, stratify=y, test_size=0.75)  # test_size stands for 3-folds

###################################################
# tune the parameters
X_tune_train, X_tune_test, y_tune_train, y_tune_test = train_test_split(X_tune, y_tune, stratify=y_tune, test_size=0.30)

# tuning with linear svm
# c_list_linear = [100, 10, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.001]
# for i in c_list_linear:
#     clf_linear = svm.SVC(kernel='linear', C=i).fit(X_tune_train, y_tune_train)
#     score = clf_linear.score(X_tune_test, y_tune_test)
#     print('C =', str(i), ', score:', str(score))
# print('After several runs, when C=0.1, outputs best score overall.')

# tuning with gaussian svm
# c_list_gaussian = [70, 60, 50, 10, 8, 5, 3, 1, 0.5, 0.1]
# gamma_list = [0.1, 0.01, 0.001, 0.0001]
# for i in c_list_gaussian:
#     clf_gaussian = svm.SVC(kernel='rbf', C=i, gamma=0.01).fit(X_tune_train, y_tune_train)
#     score = clf_gaussian.score(X_tune_test, y_tune_test)
#     print('C =', str(i), ' score:', str(score))
    # for j in gamma_list:
    #     clf_gaussian = svm.SVC(kernel='rbf', C=i, gamma=j).fit(X_tune_train, y_tune_train)
    #     score = clf_gaussian.score(X_tune_test, y_tune_test)
    #     print('C =', str(i), 'gamma =', str(j), ' score:', str(score))

###################################################
# split 3_fold into 3 folds with StratifiedKFold
skf = StratifiedKFold(n_splits=3)
# X_train = []
# X_test = []
# y_train = []
# y_test = []
precision_list_linear = []
precision_list_gaussian = []
counter = 1
for train_index, test_index in skf.split(X_3_fold, y_3_fold):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf_linear = svm.SVC(kernel='linear', C=0.1).fit(X_train, y_train)
    clf_gaussian = svm.SVC(kernel='rbf', C=5, gamma=0.01).fit(X_train, y_train)

    # calculate the accuracy for linear model
    y_linear_pred = pd.Series(clf_linear.predict(X_test), name='Prediction')
    y_linear_actu = pd.Series(y_test, name='Actual')
    df_confusion_linear = pd.crosstab(y_linear_actu, y_linear_pred)
    nii_linear = df_confusion_linear.iloc[0][0] + df_confusion_linear.iloc[1][1]
    n_linear = nii_linear + df_confusion_linear.iloc[0][1] + df_confusion_linear.iloc[1][0]
    precision_linear = nii_linear / n_linear
    precision_list_linear.append(precision_linear)

    # calculate the accuracy for gaussian model
    y_gaussian_pred = pd.Series(clf_gaussian.predict(X_test), name='Prediction')
    y_gaussian_actu = pd.Series(y_test, name='Actual')
    df_confusion_gaussian = pd.crosstab(y_gaussian_actu, y_gaussian_pred)
    nii_gaussian = df_confusion_gaussian.iloc[0][0] + df_confusion_gaussian.iloc[1][1]
    n_gaussian = nii_gaussian + df_confusion_gaussian.iloc[0][1] + df_confusion_gaussian.iloc[1][0]
    precision_gaussian = nii_gaussian/n_gaussian
    precision_list_gaussian.append(precision_gaussian)
    print('###########################################################')
    print('The', str(counter), 'fold evaluation:')
    print('Linear SVM evaluation:                     ')
    print(df_confusion_linear)
    print('precision =', str(precision_linear), '\n')
    print('Gaussian SVM evaluation:                   ')
    print(df_confusion_gaussian)
    print('precision =', str(precision_gaussian), '\n')
    counter += 1

print('###########################################################')
print('#              Linear SVM evaluation report               #')
print('###########################################################')
print('Average accuracy:', str("{:.2f}".format(mean(precision_list_linear))))
print('Standard deviation:', str("{:.2f}".format(stdev(precision_list_linear))), '\n')

print('###########################################################')
print('#              Gaussian SVM evaluation report             #')
print('###########################################################')
print('Average accuracy:', str("{:.2f}".format(mean(precision_list_gaussian))))
print('Standard deviation:', str("{:.2f}".format(stdev(precision_list_gaussian))), '\n')
