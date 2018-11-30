CSCI57300	Data Mining
Homework 4  SVM & Eclact Algorithm
Author: Jingyi Hui
Data:	11/30/2018

-----------------------------------------
List of Documents:
1. README.txt
2. JH_HW4_SVM.py
3. JH_HW4_Eclat.py
4. house-votes-84.data
5. eclat.exec (C version of eclat algorithm implementation)

-----------------------------------------
To run the program:

1. Login to Tesla and copy all the files under a directory;

2. Make the eclat and .py file executable, type:
	chmod 777 eclat
	chmod +x *.py

3. For the first question:
	3.1 type:
		./JH_HW4_SVM.py
	3.2 The program will generate 'Q1_report.txt'
	3.3 The report only contains the final results of the evaluation. More detailed information (paramaters, confusion matrix, precision) is printed in the console.

4. For the second question:
	4.1 type:
		./JH_HW4_Eclat.py
	4.2	The script will first generate file 'converted.tab' (converted dataset) as the input for eclat
	4.3 The script will then automatically call the Eclat program to generate 2 documents for future use: 
		(a) 'freq.out': Frequent itemset
		(b) 'association.out': association rule with minimum confidence 0.75 
	4.4 The script will print all the answers for question 2 in the console, and also generate file 'Q2_report.txt'. Details like paramaters, confusion matrix, precision will only be shown in the console.

5. In question 2 f and h, program only preserves rules with confidence larger than 0.75, rules with confidence equal to 0.75 are not counted. 

6. For both readability and presision, all the decimals are set to 3 or 4 digits.
