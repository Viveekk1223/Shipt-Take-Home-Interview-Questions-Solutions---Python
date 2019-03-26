# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:10:17 2019

@author: shah1
"""

#Setting directoy to the folder that contains the data 
#Importing pandas and json libraries
import pandas as pd
import json

parsing_data = pd.read_csv('InterviewData_Parsing.csv')

'''
For this question, I tried to solve it using two diferent methods.

1) I imported the files and created a JSON object of the column data so that I can use
the JSON library to splice and split the data. After doing that, I appended the values in the values array
and then added that array to my data

'''

values = []
X = True
for index,row in parsing_data.iterrows():
    data = row['data_to_parse']
    #The data is not a json object so I've removed the brackets at the end and 
    #made it a object which is a valid input for json loads function
    data = '{' + data[:-2] + '}'
    
    #Now that it is a json object, I have used json loads function to splice
    #the data and split it on ; delimiter
    splitvals = json.loads(data)['value'].split(";")
    
    #Adding the splitted values as different columns with column name as split_index
    for idx,val in enumerate(splitvals):
        if X == True:
            values.append([])
        values[idx].append(val)
    
    X = False    
        
print (len(values))        
for idx in range(len(values)):
    parsing_data['Split '+str(idx+1)] = values[idx]



'''

2) I considered it a normal string tried to split it and save it in different columns using pandas library

'''        

parsing_data_2 = pd.read_csv('InterviewData_Parsing.csv')
second_col_values = parsing_data_2['data_to_parse'].values
# split string into an array of 4 values
separate_col_values = [x.split('"')[3].split(";") for x in second_col_values]
# adds that back to the data frame
parsing_data_2['split1'] = [x[0] for x in pd.Series(separate_col_values)]
parsing_data_2['split2'] = [x[1] for x in pd.Series(separate_col_values)]
parsing_data_2['split3'] = [x[2] for x in pd.Series(separate_col_values)]
parsing_data_2['split4'] = [x[3] for x in pd.Series(separate_col_values)]



   
        
    