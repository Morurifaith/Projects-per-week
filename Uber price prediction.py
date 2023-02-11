#!/usr/bin/env python
# coding: utf-8

# # UBER PRICE PREDICTION

# Import libraries 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
from statsmodels.tools.eval_measures import rmse


# In[2]:


from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[3]:


df=pd.read_csv('C:\\Users\\ALVIN\\Desktop\\uber.csv')
df


# In[4]:


#Check for empty elements

nvc = pd.DataFrame(df.isnull().sum().sort_values(), columns=['Total Null Values'])
nvc['Percentage'] = round(nvc['Total Null Values']/df.shape[0],3)*100
print(nvc)
df.dropna(inplace=True)


# In[5]:


df.info()


# In[6]:


df.describe


# In[7]:


df.dtypes


# In[8]:


corr=df.corr(method='kendall')
corr


# In[9]:


plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True)


# In[10]:


df.describe()


# In[11]:


df.drop(columns=df.columns[0], axis=1,  inplace=True)
df.head(10)


# In[12]:


df.shape


# In[13]:


df.dtypes


# In[14]:


df.pickup_datetime=pd.to_datetime(df.pickup_datetime,errors='coerce')


# In[15]:


df.dtypes


# In[16]:


df=df.assign(hour=df.pickup_datetime.dt.hour, day=df.pickup_datetime.dt.day, month=df.pickup_datetime.dt.month, year=df.pickup_datetime.dt.year, dayofweek=df.pickup_datetime.dt.dayofweek)
df.head(10)


# In[17]:


df.nunique().sort_values()


# # UNDERSTANDING THE RELATIONSHIP BETWEEN VARIABLES AND PRICE

# In[32]:


sns.countplot(df.month,order = df['month'].value_counts().index)


# In[22]:


sns.countplot(df.hour,order = df['hour'].value_counts().index)


# In[24]:


sns.countplot(df.dayofweek,order = df['dayofweek'].value_counts().index)


# In[26]:


sns.countplot(df.passenger_count,order = df['passenger_count'].value_counts().index)


# In[27]:


sns.countplot(df.year,order = df['year'].value_counts().index)


# In[39]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'year', y = 'fare_amount', data = df, order = df['year'].value_counts().index)


# In[35]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'passenger_count', y = 'fare_amount', data = df)


# In[38]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'hour', y = 'fare_amount', data = df)


# In[ ]:




