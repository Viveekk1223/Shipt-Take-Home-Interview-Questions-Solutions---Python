# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:52:53 2019

@author: shah1
"""

#Question 5

#Setting the work directory to the folder containing the dataset
#Importing the required libraries
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_curve,auc

df = pd.read_csv("InterviewData_Activity.csv")


# The code for the model is already given and I've made the changes to model to train it on the training_data dataset
# and test the model on test_data dataset 
dummy_genders = pd.get_dummies(df['gender'], prefix = 'gender')
dummy_metro = pd.get_dummies(df['metropolitan_area'], prefix = 'metro_area')
dummy_device = pd.get_dummies(df['device_type'], prefix = 'device')

cols_to_keep = ['active', 'age']

activity_data = df[cols_to_keep].join(dummy_genders.ix[:, 'gender_M':])
activity_data = activity_data.join(dummy_metro.ix[:, 'metro_area_Birmingham':])
activity_data = activity_data.join(dummy_device.ix[:, 'device_Mobile':])

activity_data = sm.add_constant(activity_data, prepend=False)

explanatory_cols = activity_data.columns[1:]

training_data = activity_data[1:4000]
test_data = activity_data[4001:].copy()

training_logit_model = sm.GLM(training_data['active'],
training_data[explanatory_cols],
family=sm.families.Binomial())

training_result = training_logit_model.fit()

# We test our model on the test_data dataset
res = training_result.predict(test_data[explanatory_cols])

#Printing out the summary to verify if it worked
print (res.summary())

#Using sklearn, we find the Area Under the Curve (AUC) of the ROC curve 

#fpr = False Positive Rate 
#tpr = True Positive Rate
fpr,tpr,thresholds = roc_curve(test_data['active'],res)
roc_auc = auc(fpr,tpr)
print("Area under the ROC Curve : %f" % roc_auc)

# AUC = 0.617114

# We use the same method to assess the accuracy as we did for the previous question to compare the results


sum = 0
total = len(res)
for index,row in test_data.iterrows():
    data = row['active']
    if res[index]>= 0.5:
       if data == 1 :
           sum = sum + 1
    else :
        if data == 0 :
            sum = sum + 1
            
print ("Accuracy : %f" % (sum*100.0/total))

# Accuracy = 21.07%

# Again, to verify if this solution is correct, I've verified it with the sklearn model that I 
# had created in the previous question

X_train = training_data[explanatory_cols]
Y_train = training_data.active
X_test = test_data[explanatory_cols]
Y_test = test_data.active
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
y_pred = logreg.predict(X_test)
cm = metrics.confusion_matrix(Y_test, y_pred)

print ("Accuracy", metrics.accuracy_score(Y_test, y_pred))
print ("Precision", metrics.precision_score(Y_test, y_pred))
print ("Recall", metrics.recall_score(Y_test, y_pred))

'''

   Accuracy 0.2149400986610289
   Precision 0.09926769731489016
   Recall 0.9457364341085271

'''
# We get the same reults which validates my solution 

'''

The accuracy drops in this case as compared to the previous case is becuase we are fitting our model 
to a different data set than the one we trained it on. 

So we are trying to predict the value of the response variable 'active' for the test_data dataset 
from our model that was trained using the training_data dataset and it did not predict well. 

Our model was not very good to begin with having accuracy 58% so it was understandable that 
the accuracy of the test_dataset prediciton has fallen so drastically.

''' 
