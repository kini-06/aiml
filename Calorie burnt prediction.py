#!/usr/bin/env python
# coding: utf-8

# In[10]:


pip install xgboost


# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# In[12]:


calories = pd.read_csv('calories.csv')


# In[13]:


calories.head()


# In[14]:


exercise_data = pd.read_csv('exercise.csv')


# In[15]:


exercise_data.head()


# In[16]:


calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)


# In[17]:


calories_data.head()


# In[18]:


calories_data.shape


# In[19]:


calories_data.info()


# In[20]:


calories_data.isnull().sum()


# In[21]:


# get some statistical measures about the data
calories_data.describe()


# In[23]:


sns.set()


# In[24]:


# plotting the gender column in count plot
sns.countplot(calories_data['Gender'])


# In[25]:


# finding the distribution of "Age" column
sns.distplot(calories_data['Age'])


# In[26]:


# finding the distribution of "Height" column
sns.distplot(calories_data['Height'])


# In[27]:


# finding the distribution of "Weight" column
sns.distplot(calories_data['Weight'])


# In[28]:


correlation = calories_data.corr()


# In[29]:


# constructing a heatmap to understand the correlation

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# In[30]:


#converting data into numerical values
calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)


# In[31]:


calories_data.head()


# In[32]:


#seperating features and target
X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']
     


# In[33]:



print(Y)


# In[34]:


#Splitting the data into training data and Test data


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[35]:


print(X.shape, X_train.shape, X_test.shape)


# In[36]:


# loading the model
model = XGBRegressor()
     


# In[37]:


# training the model with X_train
model.fit(X_train, Y_train)


# In[38]:


test_data_prediction = model.predict(X_test)


# In[39]:


print(test_data_prediction)


# In[40]:


mae = metrics.mean_absolute_error(Y_test, test_data_prediction)


# In[41]:


print("Mean Absolute Error = ", mae)


# In[ ]:




