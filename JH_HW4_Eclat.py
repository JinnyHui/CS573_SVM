#!/usr/bin/env python3
# Jingyi hui 11/24/2018
# CSCI573 DATAMINING
# HOMEWORK 4 Eclat
# all the transaction files are comma separated txt files

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import svm
from statistics import mean, stdev
import numpy as np
import sys


##################################################
# convert the dataset into an item-set dataset
f = open('house-votes-84.data', 'r')
o = open('converted.tab', 'w')
lines = f.readlines()
for line in lines:
    strpline = line.rstrip()
    arr = strpline.split(',')
    newline = []
    for i in range(len(arr)):
        if arr[i] == 'y':
            newline.append(str(i))
    if arr[0] == 'republican':
        newline.append(str(100))
    else:
        newline.append(str(200))
    # print(*newline, sep=' ')
    myline = '\t'.join(newline)
    o.write(myline + '\n')
o.close()

###################################################
# run the eclat algorithm with 20% support with descending support
# './eclat -s20 converted.tab freq.out'
# generate association rule
# './eclat -tr -s20 -v" (%c)" converted.tab association.out

###################################################
# read in the frequent set file, load it into a dict
file_in = open('freq.out', 'r')
lines = file_in.readlines()
item_dict = {}
for line in lines:
    strpline = line.rstrip()
    arr = strpline.split(' ')
    key_tuple = tuple(map(int, arr[:-1]))
    item_dict[key_tuple] = int(arr[-1].strip('()'))
file_out = open('Q2_report.txt', 'w')

###################################################
# a. Run the itemset mining algorithm with 20% support. How many frequent itemsets are there
ans_a = "Q2a: The number of frequent itemsets is: " + str(len(item_dict))
print(ans_a)
file_out.write(ans_a + '\n')

###################################################
# b. Write top 10 itemsets (in terms of highest support value)
top_10 = sorted(item_dict.items(), key=lambda x: -x[1])[:10]
counter = 1
file_out.write('\nQ2b: Top 10 itemsets are:\n')
print('\nQ2b: Top 10 itemsets are:')
for item in top_10:
    freq_set, freq = item
    ans_b = 'Frequent set NO.%-3s:%-10s Frequency (support): %s' % (str(counter), str(freq_set), str(freq))
    print(ans_b)
    file_out.write(ans_b + '\n')
    counter += 1

###################################################
# c. How many frequent itemsets have 100 as part of itemsets
# print('\nQ2c: Frequent itemsets have 100 are:')
counter_100 = 0
for key, value in item_dict.items():
    if 100 in key:
        counter_100 += 1
        # print(key)
ans_c = '\nQ2c: The number of frequent itemsets have 100 is: ' + str(counter_100)
print(ans_c)
file_out.write(ans_c)

###################################################
# d. How many frequent itemsets have 200 as part of itemsets
# print('\nQ2d: Frequent itemsets have 200 are:')
counter_200 = 0
for key, value in item_dict.items():
    if 200 in key:
        counter_200 += 1
        # print(key)
ans_d = '\nQ2d: The number of frequent itemsets have 200 is: ' + str(counter_200)
print(ans_d)
file_out.write('\n' + ans_d)

###################################################
# read in the association rule file, store into a list
file_rule = open('association.out', 'r')
lines = file_rule.readlines()
rule_dict = {}
for line in lines:
    strpline = line.rstrip()
    arr = strpline.split(' ')
    arr[-1] = float("{0:.4f}".format(float(arr[-1].strip('()'))))
    body_tuple = tuple(arr[2:-1])
    key_tuple = (int(arr[0]), body_tuple)
    value = arr[-1]
    rule_dict[key_tuple] = value
feature_rule = {}

###################################################
# construct dictionary of rules with head 100, 200
dict_100 = {}
dict_200 = {}
for key, value in rule_dict.items():
    if key[0] == 100:
        dict_100[key] = value
    elif key[0] == 200:
        dict_200[key] = value

###################################################
# e. Write top 10 association rules where the rule’s head is 100
print('\nQ2e: Top 10 association rules where the rule’s head is 100: ')
file_out.write('\n' + '\nQ2e: Top 10 association rules where the rule’s head is 100:')
ordered_100 = sorted(dict_100.items(), key=lambda x: -x[1])
count_100_rule = 0
ordered_100_counter = 0
for i in ordered_100:
    # print(i)
    ordered_100_counter += 1
    count_100_rule += 1
    rule, confi = i
    body = ', '.join(list(rule[1]))
    rule_string = str(rule[0]) + ' <- ' + body
    ans_e = 'Rule NO.%-3s: %-26s Confidence: %s' % (str(count_100_rule), rule_string, str(confi))
    print(ans_e)
    file_out.write('\n' + ans_e)
    if ordered_100_counter > 9:
        break

###################################################
# f. List rules with head 100 which the confidence value is more than 75%
print('\nQ2f: Rules with head 100 with confidence higher than 75% are: ')
print('(confidence equal to 0.75 is not included, ordered by confidence)')
file_out.write('\n\nQ2f: Rules with head 100 with confidence higher than 75% are: ')
file_out.write('\n(confidence equal to 0.75 is not included, ordered by confidence)')
count_100_75 = 0
for i in ordered_100:
    rule, confi = i
    if confi > 0.75:
        count_100_75 += 1
        head, body = rule
        feature_rule[count_100_75 - 1] = body
        body_string = ', '.join(list(body))
        rule_string = str(head) + ' <- ' + body_string
        ans_f = 'Rule NO.%-3s: %-26s Confidence: %s' % (str(count_100_75), rule_string, str(confi))
        print(ans_f)
        file_out.write('\n' + ans_f)
print('Total number is:', str(count_100_75))
file_out.write('\nTotal number is:' + str(count_100_75))

###################################################
# g. Write top 10 association rules where the rule’s head is 200
print('\nQ2g: Top 10 association rules where the rule’s head is 200: ')
file_out.write('\n\nQ2g: Top 10 association rules where the rule’s head is 200:')
ordered_200 = sorted(dict_200.items(), key=lambda x: -x[1])
ordered_200_counter = 0
for i in ordered_200:
    # print(i)
    ordered_200_counter += 1
    rule, confi = i
    body = ', '.join(list(rule[1]))
    rule_string = str(rule[0]) + ' <- ' + body
    ans_g = 'Rule NO.%-3s: %-26s Confidence: %s' % (str(ordered_200_counter), rule_string, str(confi))
    print(ans_g)
    file_out.write('\n' + ans_g)
    if ordered_200_counter > 9:
        break

###################################################
# h. List rules with head 200 which the confidence value is more than 75%
print('\nQ2h: Rules with head 200 with confidence higher than 75% are: ')
print('(confidence equal to 0.75 is not included, ordered by confidence)')
file_out.write('\n\nQ2h: Rules with head 200 with confidence higher than 75% are: ')
file_out.write('\n(confidence equal to 0.75 is not included, ordered by confidence)')
count_200_75 = 0
for i in ordered_200:
    rule, confi = i
    if confi > 0.75:
        count_200_75 += 1
        head, body = rule
        feature_rule[count_100_75 + count_200_75 - 1] = body
        body_string = ', '.join(list(body))
        rule_string = str(head) + ' <- ' + body_string
        ans_h = 'Rule NO.%-3s: %-26s Confidence: %s' % (str(count_200_75), rule_string, str(confi))
        print(ans_h)
        file_out.write('\n' + ans_h)
    # else:
    #     print('There is a 75%', key, value)
print('Total number is:', str(count_200_75))
file_out.write('\nTotal number is:' + str(count_200_75))

###################################################
# read each converted transaction and construct the new dataset
trans = open('converted.tab', 'r')
new_data = open('new_data.txt', 'w')
lines = trans.readlines()
line_counter = 0
svm_dataset = np.zeros(shape=(len(lines), len(feature_rule) + 1))
for line in lines:
    strpline = line.rstrip()
    # print(strpline)
    arr = strpline.split('\t')
    body_set = set(arr[:-1])
    for key, value in feature_rule.items():
        feature = set(value)
        if feature.issubset(body_set):
            svm_dataset[line_counter][key] = int(1)
    svm_dataset[line_counter][-1] = int(arr[-1])
    line_counter += 1

###################################################
# i. soft-margin SVM on rules with more than 75% confidence
# split the dateset into data and label
X = svm_dataset[:, :-1]
y = svm_dataset[:, -1]
X_tune, X_3_fold, y_tune, y_3_fold = train_test_split(X, y, stratify=y, test_size=0.75, random_state=50)
# c_list_linear = [100, 80, 60, 40, 20, 1]
# c_list_gaussian = [80, 75, 70, 65, 60, 55, 50, 45, 40]
# gamma_list_gaussian = [0.01]
skf = StratifiedKFold(n_splits=3, random_state=100)
C_linear_list = [1, 0.3, 5]
C_gaussian_list = [50, 10, 60]
gamma_list = [0.01, 0.01, 0.01]
precision_list_linear = []
precision_list_gaussian = []
counter = 0
for train_index, test_index in skf.split(X_3_fold, y_3_fold):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # linear parameter tuning
    # for i in c_list_linear:
    #     clf_linear = svm.SVC(kernel='linear', C=i).fit(X_train, y_train)
    #     score = clf_linear.score(X_tune, y_tune)
    #     print('C =', str(i), ', score:', str(score))
    clf_linear = svm.SVC(kernel='linear', C=C_linear_list[counter]).fit(X_train, y_train)
    # Gaussian parameter tuning
#     for i in c_list_gaussian:
#         for j in gamma_list_gaussian:
#             clf_gaussian = svm.SVC(kernel='rbf', C=i, gamma=j).fit(X_train, y_train)
#             score = clf_gaussian.score(X_tune, y_tune)
#             print('C =', str(i), 'gamma =', str(j), 'score:', str(score))
    clf_gaussian = svm.SVC(kernel='rbf', C=C_gaussian_list[counter], gamma=gamma_list[counter]).fit(X_train, y_train)
# calculate the accuracy for linear model
    y_linear_pred = pd.Series(clf_linear.predict(X_test), name='Prediction')
    y_linear_actu = pd.Series(y_test, name='Actual')
    df_confusion_linear = pd.crosstab(y_linear_actu, y_linear_pred)
    nii_linear = df_confusion_linear.iloc[0, 0] + df_confusion_linear.iloc[1, 1]
    n_linear = nii_linear + df_confusion_linear.iloc[0, 1] + df_confusion_linear.iloc[1, 0]
    precision_linear = nii_linear / n_linear
    precision_list_linear.append(precision_linear)

    # calculate the accuracy for gaussian model
    y_gaussian_pred = pd.Series(clf_gaussian.predict(X_test), name='Prediction')
    y_gaussian_actu = pd.Series(y_test, name='Actual')
    df_confusion_gaussian = pd.crosstab(y_gaussian_actu, y_gaussian_pred)
    nii_gaussian = df_confusion_gaussian.iloc[0, 0] + df_confusion_gaussian.iloc[1, 1]
    n_gaussian = nii_gaussian + df_confusion_gaussian.iloc[0, 1] + df_confusion_gaussian.iloc[1, 0]
    precision_gaussian = nii_gaussian/n_gaussian
    precision_list_gaussian.append(precision_gaussian)

    print('################################')
    print('    The NO.', str(counter + 1), '3-fold evaluation')
    print('Linear SVM evaluation:                     ')
    print(df_confusion_linear)
    print('Best parameter C = ', str(C_linear_list[counter]))
    print('precision =', str("{:.3f}".format(precision_linear)), '\n')
    print('Gaussian SVM evaluation:                   ')
    print(df_confusion_gaussian)
    print('Best parameter C = ', str(C_gaussian_list[counter]), 'gamma =', str(gamma_list[counter]))
    print('precision =', str("{:.3f}".format(precision_gaussian)), '\n')
    counter += 1

print('\nQ2i: SVM 3-fold evaluation report:')
separator = '###########################################################'
avg_acc_linear_str = str("{:.3f}".format(mean(precision_list_linear)))
avg_dev_linear_str = str("{:.3f}".format(stdev(precision_list_linear)))
print('\n' + separator + '\n#              Linear SVM evaluation report               #' + '\n' + separator)
print('Average accuracy:' + avg_acc_linear_str)
print('Standard deviation:' + avg_dev_linear_str + '\n')
avg_acc_gaussian_str = str("{:.3f}".format(mean(precision_list_gaussian)))
avg_dev_gaussian_str = str("{:.3f}".format(stdev(precision_list_gaussian)))
print(separator + '\n#              Gaussian SVM evaluation report             #' + '\n' + separator)
print('Average accuracy:' + avg_acc_gaussian_str)
print('Standard deviation:' + avg_dev_gaussian_str + '\n')

# output to txt
file_out.write('\n\nQ2i: SVM 3-fold evaluation report:')
file_out.write('\n' + separator + '\n#              Linear SVM evaluation report               #' + '\n' + separator)
file_out.write('\nAverage accuracy:' + avg_acc_linear_str)
file_out.write('\nStandard deviation:' + avg_dev_linear_str)
file_out.write('\n\n' + separator + '\n#              Gaussian SVM evaluation report             #' + '\n' + separator)
file_out.write('\nAverage accuracy:' + avg_acc_gaussian_str)
file_out.write('\nStandard deviation:' + avg_dev_gaussian_str)

file_out.close()
