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
X_tune, X_3_fold, y_tune, y_3_fold = train_test_split(X, y, stratify=y, test_size=0.75, random_state=100)  # test_size stands for 3-folds

###################################################
# set the tuning parameters
c_list_linear = [100, 10, 1, 0.5, 0.4, 0.3, 0.2, 0.1, 0.001]
c_list_gaussian = [70, 60, 50, 10, 8, 5, 3, 1, 0.5, 0.1]
gamma_list_gaussian = [0.1, 0.01, 0.001, 0.0001]

###################################################
# split 3_fold into 3 folds with StratifiedKFold
skf = StratifiedKFold(n_splits=3, random_state=100)
C_linear_list = [0.1, 0.1, 0.1]
C_gaussian_list = [5, 8, 10]
gamma_list = [0.1, 0.01, 0.1]
precision_list_linear = []
precision_list_gaussian = []
counter = 1
for train_index, test_index in skf.split(X_3_fold, y_3_fold):
    X_train, X_test = X_3_fold[train_index], X_3_fold[test_index]
    y_train, y_test = y_3_fold[train_index], y_fold[test_index]

    # linear parameter tuning
    # for i in c_list_linear:
    #     clf_linear = svm.SVC(kernel='linear', C=i).fit(X_train, y_train)
    #     score = clf_linear.score(X_tune, y_tune)
    #     print('C =', str(i), ', score:', str(score))
    clf_linear = svm.SVC(kernel='linear', C=C_linear_list[counter-1]).fit(X_train, y_train)

    # Gaussian parameter tuning
    # for i in c_list_gaussian:
    #     for j in gamma_list_gaussian:
    #         clf_gaussian = svm.SVC(kernel='rbf', C=i, gamma=j).fit(X_train, y_train)
    #         score = clf_gaussian.score(X_tune, y_tune)
    #         print('C =', str(i), 'gamma =', str(j), 'score:', str(score))

    clf_gaussian = svm.SVC(kernel='rbf', C=C_gaussian_list[counter-1], gamma=gamma_list[counter-1]).fit(X_train, y_train)

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
    print('################################')
    print('    The NO.', str(counter), '3-fold evaluation')
    print('Linear SVM evaluation:                     ')
    print(df_confusion_linear)
    print('Best parameter C = ', str(C_linear_list[counter-1]))
    print('precision =', str("{:.3f}".format(precision_linear)), '\n')
    print('Gaussian SVM evaluation:                   ')
    print(df_confusion_gaussian)
    print('Best parameter C = ', str(C_gaussian_list[counter - 1]), 'gamma =', str(gamma_list[counter-1]))
    print('precision =', str("{:.3f}".format(precision_gaussian)), '\n')
    counter += 1

report_out = open('Q1_report.txt', 'w')
separator = '###########################################################'
avg_acc_linear_str = str("{:.3f}".format(mean(precision_list_linear)))
avg_dev_linear_str = str("{:.3f}".format(stdev(precision_list_linear)))
print(separator + '\n#              Linear SVM evaluation report               #' + '\n' + separator)
print('Average accuracy:' + avg_acc_linear_str)
print('Standard deviation:' + avg_dev_linear_str + '\n')
avg_acc_gaussian_str = str("{:.3f}".format(mean(precision_list_gaussian)))
avg_dev_gaussian_str = str("{:.3f}".format(stdev(precision_list_gaussian)))
print(separator + '\n#              Gaussian SVM evaluation report             #' + '\n' + separator)
print('Average accuracy:' + avg_acc_gaussian_str)
print('Standard deviation:' + avg_dev_gaussian_str + '\n')

# output to txt
report_out.write(separator + '\n#              Linear SVM evaluation report               #' + '\n' + separator)
report_out.write('\nAverage accuracy:' + avg_acc_linear_str)
report_out.write('\nStandard deviation:' + avg_dev_linear_str)
report_out.write('\n\n' + separator + '\n#              Gaussian SVM evaluation report             #' + '\n' + separator)
report_out.write('\nAverage accuracy:' + avg_acc_gaussian_str)
report_out.write('\nStandard deviation:' + avg_dev_gaussian_str)

report_out.close()
