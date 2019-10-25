#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''





'''


import os
import sys
import pandas as pd
import numpy as np
#import seaborn as sns
#import boto3
#import botocore


# In[3]:



#BUCKET_NAME = 'pcadsassessment2' # replace with your bucket name
#KEY = 'parking_citations.corrupted.csv' # replace with your object key

#s3 = boto3.resource('s3')

#s3.Bucket(BUCKET_NAME).download_file(KEY, 'input/parking_citations.corrupted.csv')


# In[7]:


#def download_file_with_resource(bucket_name, key, local_path):
#    s3 = boto3.resource('s3')
#    s3.Bucket(bucket_name).download_file(key, local_path)
#    print('Downloaded File with boto3 resource')

#bucket_name = 'pcadsassessment2'
#key = 'parking_citations.corrupted.csv' 
#local_path = 'C:\\Users\\bijan\\Traffic_Citations_Grainger'

#download_file_with_resource(bucket_name, key, local_path)


# In[8]:


#s3_client = boto3.client('s3') #low-level functional API

#s3_resource = boto3.resource('s3') #high-level object-oriented API

#my_bucket = s3_resource.Bucket('pcadsassessment') #subsitute this for your s3 bucket name.

#data_path =  'parking_citations.corrupted.csv'

#obj = s3_client.get_object(Bucket = 'my-bucket', Key = data_path)

#grid_sizes = pd.read_csv(obj['Body'])


# In[4]:


df_parking_citations = pd.DataFrame(data = pd.read_csv('input/parking_citations.corrupted.csv'))

df_parking_citations


# In[5]:


# Data Manipulations

df_parking_citations

# Drop the datapoint with Null value in the following 10 columns with number of null values less than 40'000 
# Effecting less than 1% of the whole training and validation data points
df_parking_citations = df_parking_citations.dropna(     axis = 'index', 
                                                        subset = 
                                                                ['Issue time', 
                                                                 'RP State Plate', 
                                                                 'Body Style', 
                                                                 'Color', 
                                                                 'Location',
                                                                 'Route',
                                                                 'Agency', 
                                                                 'Violation Description', 
                                                                 'Fine amount', 
                                                                 'Latitude', 
                                                                 'Longitude']
                                                  )


# Drop the whole column for the following columns due to the high number of null value and noises especially  'VIN'
df_parking_citations = df_parking_citations.drop(   columns = 
                                                                 ['Meter Id',
                                                                  'Marked Time',
                                                                  'VIN'                                            
                                                                 ]
                                                )

# Filling the Plate Expiry Date with the constant value of 202012 assuming the States have 3 years expiry date program
df_parking_citations['Plate Expiry Date'].fillna(value= 202012.00, inplace= True )

# Some Ticket number contains a letter D at the end of them which 
# shows that those ticket number were deleted or meant to be deleted
# Remove those letter for the porpuse of this analytics
df_parking_citations['Ticket number'] = df_parking_citations['Ticket number'].astype(str).str.replace('D', '')


# Change the data type from object to time stamp for Issue Date
# TODO: with Issue Date we can identify day of week, weekend or weekdays, and holiday to expand 
df_parking_citations['Issue Date'] =  pd.to_datetime(df_parking_citations['Issue Date'], format='%Y-%m-%dT%H:%M:%S')

# Identify the ticket issuance time of Day: AM: Morning or PM: Afternoon 
df_parking_citations['Time of Day'] =  [ 'AM' if x < 1200 else 'PM' for x in df_parking_citations['Issue time']]


#df_parking_citations


# In[11]:


# Categorical boolean mask
categorical_feature_mask = df_parking_citations.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = df_parking_citations.columns[categorical_feature_mask].tolist()


# In[12]:


categorical_cols = ['RP State Plate', 'Body Style' , 'Color', 'Location', 'Route', 'Violation code' , 'Violation Description', 'Time of Day']


# In[13]:


#Labeling/Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_encode = df_parking_citations

df_encode[categorical_cols] = df_encode[categorical_cols].apply(lambda col: le.fit_transform(col))

df_encode


# In[ ]:


#Normalization of the 'Fine amount'

x= df_parking_citations['Fine amount']

# Normalization to using Min-Max Scaler
df_encode['Fine amount'] = (x-x.min())/(x.max()-x.min())


# In[17]:


#TODO: Inverse transform 
#Map the Encoders to Dictionary for inverse transform 
#from sklearn.preprocessing import LabelEncoder


#le = LabelEncoder()
#from collections import defaultdict
#d = defaultdict(LabelEncoder)


# Encoding the variable
#fit = df_encode.apply(lambda x: d[x.name].fit_transform(x))

# Inverse the encoded
#fit.apply(lambda x: d[x.name].inverse_transform(x))

# Using the dictionary to label future data
#encoded.apply(lambda x: d[x.name].transform(x))


# In[18]:


# Split the dataframe to two dataframes one for train-test and the other one for prediction based on Make Column
df_train_test = df_encode.dropna(axis = 'index', subset = ['Make'])

df_predict = df_encode[df_encode['Make'].isnull()]


# In[19]:


#Identifying the top 25 common Make using groupby and aggregation methods

df_common_make = df_train_test[['Ticket number','Make']].groupby(['Make'])                                                             .agg('count')                                                             .sort_values('Ticket number', ascending = False)                                                             .head(25)                                                            .reset_index()                                                            
l_common_make = df_common_make['Make'].tolist()

df_train_test ['Common make'] = [1 if x in l_common_make else 0 for x in df_train_test['Make']]


df_predict['Common make'] = np.nan


# In[20]:



X_train_test = df_train_test.drop(['Issue Date','Make', 'Common make'],axis =1)
y_train_test = df_train_test['Common make']


# In[21]:


X_predict = df_predict.drop(['Issue Date','Make', 'Common make'],axis =1)
y_predict = df_predict['Common make']


# In[22]:


## Split data into training and testing sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_test,y_train_test, random_state = 200, test_size=0.2)


# In[ ]:


# KNN Model

from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 100)
# Fit the classifier to the data
knn.fit(X_train, y_train)


# In[ ]:


#check accuracy of our model on the train data
round(knn.score(X_train, y_train),4)


# In[ ]:


# Checking the accuracy of our model based on the training dataset and labeled feature result in 92.84% accuracy
# State-of-the-art Convolutional Neural Networks may achieve about 95% but KNN yeild an acceptable level of accuracy


# In[ ]:


#check accuracy of our model on the test data
round(knn.score(X_test, y_test),4)


# In[ ]:


# The accuracy of the trained modeled decreased to 89.84% due to the selected number of neighbors = 300 which
# can be improved by utilizing higher number of neighborhoods k =1000 can be used as a commonly used hyper-parameter for larger dataset
# This accuracy would also improve by utilizing K-fold Cross Validation as following


# In[9]:


# Checking the K-fold Cross Validation instead of train-test_split method
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5, random_state=None)

#train model with cv of 5 
cv_scores = cross_val_score(knn, X_train_test, y_train_test, cv=kfold)
#print each cv score (accuracy) and average them
print(round(cv_scores,4))
print('cv_scores mean:{}'.format(round(np.mean(cv_scores),4)))


# In[ ]:


# Cross Validation will increase the accuracy of the model to 89.88%


# In[130]:


# Predicting the target feature based on the text dataset 
yte_predict = knn.predict(X_test)


# In[160]:


# Calculating the accuracy of prediction on the test based on the comparison to the y_test
round(np.mean(yte_predict == y_test),4)


# In[ ]:


# Yielded accuracy is falling in the reseanable range from test accuracy  


# In[173]:


#Calculate the probability that a vehicle is made by one of the top 25 common manufacturers on test set

proba_test = knn.predict_proba(X_test)


# In[226]:


# the probability of top 25 common make on the test set
Common_make_proba_test = [round(l.tolist()[1],4) for l in proba_test]
Common_make_proba_test


# In[162]:


round(yte_predict.mean(),4)*100


# In[169]:


list(yte_predict)


# In[118]:


# predict the deleted labels of Make based on k - Nearest Neighbor  (KNN) Classifier

y_predict = knn.predict(X_predict)


# In[242]:


#Calculate the probability that a vehicle is made by one of the top 25 common manufacturers on predict set

proba_predict = knn.predict_proba(X_predict)


# In[243]:


proba_predict


# In[244]:


# the probability of top 25 common make on the test set
Common_make_proba_predict = [round(l.tolist()[1],4) for l in proba_predict]
Common_make_proba_predict


# In[246]:


y_predict.tolist()


# In[247]:


df_predict['Common make'] = y_predict.tolist()
df_predict['Common make Probability'] = Common_make_proba_predict


# In[121]:


#Calculate the mean probability that a vehicle is made by one of the top 25 common manufacturers 

y_predict.mean()


# In[154]:


# Calculating the Percent of Common Make on Average
round(df_train_test['Common make'].mean(),3)


# In[263]:


df_predict.head(15)


# In[273]:


X_predict.loc[13]


# In[277]:


# Test the prediction process of the single new observation with two samples
X1_predict = X_predict.loc[[13]]    #X_predict.loc[[13]] #X_predict.sample(n=2, random_state=1)
X1_predict


# In[278]:


# Prediction of new observation
y1_predict = knn.predict(X1_predict)
y1_predict 


# In[279]:


# calculating the probability of the new observation being among top 25 common Make
proba_predict1 = knn.predict_proba(X1_predict)


# In[ ]:


Common_make_proba_predict1 = proba_predict1.tolist()

