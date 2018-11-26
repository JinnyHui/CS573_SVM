#!/usr/bin/env python3
# Jingyi hui 11/24/2018
# CSCI573 DATAMINING
# HOMEWORK 4 Eclat
# all the transaction files are comma separated txt files

import pandas as pd
import numpy as np
import csv
import sys


###################################################
# convert the dataset into an itemset dataset
f = open('house-votes-84.data', 'r')
o = open('converted.txt', 'w')
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
    print(*newline, sep=',')
    myline = '\t'.join(newline)
    o.write(myline + '\n')
o.close()

