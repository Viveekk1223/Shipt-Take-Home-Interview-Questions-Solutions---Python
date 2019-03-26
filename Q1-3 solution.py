# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 12:51:12 2019

@author: shah1
"""

# Shipt - Interveiw Exercise (Take Home Test)

#Setting the work directory to folder containing the dataset

#Question 1
# importing pandas library

import pandas as pd
import matlpotlib.pyplot as plt 

Cost = pd.read_csv("InterviewData_Cost.csv")
Revenue = pd.read_csv("InterviewData_Rev.csv")

#Outer Join on Cost and Revenue 
Q1_solution = pd.merge(Cost, Revenue, how='outer' ,on = ['date', 'source_id'])  

# Question 2

# We need data from the Cost dataframe where the corresponding date is not present in the Revenue dataframe 
# Speaking in terms of joins, we essentially want to subtract the inner join rows from the left outer join on Cost dataframe
# We could've also removed all the Reveue rows from our solution from the 1st question. 

#Left Outer join on Cost
Left_Outer_Join = pd.merge(Cost, Revenue, how='left', on = ['date', 'source_id'] , indicator=True);

#We use indicator argument to show which dataframe the value is coming from for further analysis

#Removing the rows which are common for both i.e. the inner join values
Q2_solution = Left_Outer_Join[Left_Outer_Join['_merge']=='left_only']

#Dropping the merge column
Q2_solution=Q2_solution.drop(columns = ['_merge'],axis = 1)

# Question 3 

# Using our question one solution, I grouped them by source ID and added the values for Cost and Revenue columns 
Groupedby_df = Q1_solution.groupby('source_id').sum()

# Sorting the Revenue column in descending order
Sorted_df = Groupedby_df.sort_values(by = 'revenue', ascending = False)

#Selecting the first 4 rows with highest revenue
Top4_df = Sorted_df.ix[0:4,:]

#There are multiple ways we can visualize the results, bar charts, scatter plot, scatter plot of the Revenue/Cost ratio
#I've tried the scatter plot of Cost v/s Revenue below 

plt.figure(figsize = (10,8))
for i,txt in enumerate(Top4_df.index):
    x = Top4_df.cost[i]
    y = Top4_df.revenue[i]
    plt.scatter(x,y,marker= 'x', color = 'red')
    plt.text(x+0.3, y+0.3 ,txt, fontsize = 9)
    
plt.xlabel('Cost')
plt.ylabel('Revenue')
plt.title('Cost v/s Revenue')    
plt.show()





