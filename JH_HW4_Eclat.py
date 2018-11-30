#!/usr/bin/env python3
# Jingyi hui 11/24/2018
# CSCI573 DATAMINING
# HOMEWORK 4 Eclat
# all the transaction files are comma separated txt files

import pandas as pd
import numpy as np
import csv
import sys


# def AssociationRule(rule_list, confi):
#     """
#     generate all the association
#     :param rule_list:
#     :param confi:
#     :return:
#     """
#     return association_list

###################################################
# convert the dataset into an itemset dataset
# f = open('house-votes-84.data', 'r')
# o = open('converted.tab', 'w')
# lines = f.readlines()
# for line in lines:
#     strpline = line.rstrip()
#     arr = strpline.split(',')
#     newline = []
#     for i in range(len(arr)):
#         if arr[i] == 'y':
#             newline.append(str(i))
#     if arr[0] == 'republican':
#         newline.append(str(100))
#     else:
#         newline.append(str(200))
#     print(*newline, sep=' ')
#     myline = '\t'.join(newline)
#     o.write(myline + '\n')
# o.close()

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
    ans_b = 'Frequent set NO.%-3s:%-10s Frequency: %s' % (str(counter), str(freq_set), str(freq))
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
    arr[-1] = float("{0:.3f}".format(float(arr[-1].strip('()'))))
    body_tuple = tuple(arr[2:-1])
    key_tuple = (int(arr[0]), body_tuple)
    value = arr[-1]
    rule_dict[key_tuple] = value

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
top_10_100 = sorted(dict_100.items(), key=lambda x: -x[1])[:10]
count_100_rule = 0
for i in top_10_100:
    # print(i)
    count_100_rule += 1
    rule, confi = i
    body = ', '.join(list(rule[1]))
    rule_string = str(rule[0]) + ' <- ' + body
    ans_e = 'Rule NO.%-3s: %-26s Confidence: %s' % (str(count_100_rule), rule_string, str(confi))
    print(ans_e)
    file_out.write('\n' + ans_e)

###################################################
# f. List rules with head 100 which the confidence value is more than 75%
print('\nQ2f: Rules with head 100 with confidence higher than 75% are: ')
file_out.write('\n\nQ2f: Rules with head 100 with confidence higher than 75% are: ')
rule_100_75 = {}
count_100_75 = 0
for key, value in dict_100.items():
    if value > 0.75:
        count_100_75 += 1
        body = ', '.join(list(key[1]))
        rule_string = str(key[0]) + ' <- ' + body
        ans_f = 'Rule NO.%-3s: %-26s Confidence: %s' % (str(count_100_75), rule_string, str(value))
        print(ans_f)
        file_out.write('\n' + ans_f)
print('Total number is:', str(count_100_75))
file_out.write('\nTotal number is:' + str(count_100_75))

###################################################
# g. Write top 10 association rules where the rule’s head is 200
print('\nQ2g: Top 10 association rules where the rule’s head is 200: ')
file_out.write('\n\nQ2g: Top 10 association rules where the rule’s head is 200:')
top_10_200 = sorted(dict_200.items(), key=lambda x: -x[1])[:10]
count_200_rule = 0
for i in top_10_200:
    # print(i)
    count_200_rule += 1
    rule, confi = i
    body = ', '.join(list(rule[1]))
    rule_string = str(rule[0]) + ' <- ' + body
    ans_g = 'Rule NO.%-3s: %-26s Confidence: %s' % (str(count_200_rule), rule_string, str(confi))
    print(ans_g)
    file_out.write('\n' + ans_g)

###################################################
# h. List rules with head 200 which the confidence value is more than 75%
print('\nQ2h: Rules with head 200 with confidence higher than 75% are: ')
file_out.write('\n\nQ2h: Rules with head 200 with confidence higher than 75% are: ')
rule_200_75 = {}
count_200_75 = 0
for key, value in dict_200.items():
    if value > 0.75:
        count_200_75 += 1
        body = ', '.join(list(key[1]))
        rule_string = str(key[0]) + ' <- ' + body
        ans_h = 'Rule NO.%-3s: %-26s Confidence: %s' % (str(count_200_75), rule_string, str(value))
        print(ans_h)
        file_out.write('\n' + ans_h)
    # else:
    #     print('There is a 75%')
print('Total number is:', str(count_200_75))
file_out.write('\nTotal number is:' + str(count_200_75))

###################################################
# construct the new dataset


###################################################
# i. soft-margin SVM on rules with more than 75% confidence



file_out.close()
