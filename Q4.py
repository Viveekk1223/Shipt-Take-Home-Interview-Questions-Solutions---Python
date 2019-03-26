# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 15:27:45 2019

@author: shah1
"""

#Question 4 

#Setting the work directory to the folder containing the dataset
#Importing the required libraries
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve,auc

df = pd.read_csv("InterviewData_Activity.csv")

#The code for the GLM model was given and the model is created using the activity_data dataset

dummy_genders = pd.get_dummies(df['gender'], prefix = 'gender')
dummy_metro = pd.get_dummies(df['metropolitan_area'], prefix = 'metro_area')
dummy_device = pd.get_dummies(df['device_type'], prefix = 'device')

cols_to_keep = ['active', 'age']

activity_data = df[cols_to_keep].join(dummy_genders.ix[:, 'gender_M':])
activity_data = activity_data.join(dummy_metro.ix[:, 'metro_area_Birmingham':])
activity_data = activity_data.join(dummy_device.ix[:, 'device_Mobile':])

activity_data = sm.add_constant(activity_data, prepend=False)

explanatory_cols = activity_data.columns[1:]

full_logit_model = sm.GLM(activity_data['active'],
activity_data[explanatory_cols],
family=sm.families.Binomial())

result = full_logit_model.fit()

#Printing out the summary to validate our model 
print (result.summary())

#Using sklearn, we find the Area Under the Curve (AUC) of the ROC curve 

#fpr = False Positive Rate 
#tpr = True Positive Rate
res = result.predict(activity_data[explanatory_cols])
fpr,tpr,thresholds = roc_curve(activity_data['active'],res)
roc_auc = auc(fpr,tpr)
print("Area under the ROC Curve : %f" % roc_auc)

#AUC = 0.613585

'''

There are different ways we assess the accuracy of a GLM model
By checking specificity, sensitivity, precision, recall or the accuracy score.

For this question I'm using the accuracy score to assess the model 

We are defining accuracy as (True Positives + True Negatives)/Total observations
The following code calculates that value for us.

'''

sum = 0
total = len(res)

for index,row in activity_data.iterrows():
    data = row['active']
    if res[index]>= 0.5:
       if data == 1 :
           sum = sum + 1
    else :
        if data == 0 :
            sum = sum + 1
            
print ("Accuracy : %f" % (sum*100.0/total)) 

'''
We get accuracy as 58.062% which is not great, I am not sure what exactly is the problem,
There could be multiple reasons for it, either there are some non-linear factors in our model 
and Logistic regression being a linear classification tool, cannot account for them, maybe we can 
try different non linear methods like SVM to check if they give a better result.

To verify that my approach is correct, I created the same model using sklearn library and got the same results
I've added to code for it below for your consideration.           

'''

X = activity_data[explanatory_cols]
Y = activity_data.active
logreg = LogisticRegression()
logreg.fit(X,Y)
y_pred = logreg.predict(X)
cm = metrics.confusion_matrix(Y, y_pred)

print ("Accuracy", metrics.accuracy_score(Y, y_pred))
print ("Precision", metrics.precision_score(Y, y_pred))
print ("Recall", metrics.recall_score(Y, y_pred))

'''
These are the results from the above code which match with my solution

      Accuracy 0.5800738007380074
      Precision 0.5667924528301886
      Recall 0.5711026615969582

'''
 

            
