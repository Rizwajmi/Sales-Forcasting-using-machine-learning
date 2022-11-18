#!/usr/bin/env python
# coding: utf-8

# # Sales forcasting using walmart dataset.

# In[1]:


#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.special import boxcox1p
import seaborn as sns


# In[34]:


import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter


# In[2]:


#Getting data
features=pd.read_csv("features.csv")
store=pd.read_csv("stores.csv")
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


# In[3]:


#From train.csv, taking groupby Store, Date and sum Weekly_Sales.
#reseting train
train=train.groupby(['Store','Date'])['Weekly_Sales'].sum()
train=train.reset_index()
train.head(10)


# In[39]:


#Merging train and features data by inner join.
#merging train and feature
data=pd.merge(train,features,on=['Store','Date'],how='inner')
data.head(8)


# In[41]:


data.describe()


# In[6]:


#sorting the data by date.
#sorting values of Data
data=data.sort_values(by='Date')
data.head(10)


# In[7]:


#Analyzing the data
#Here, we see different methods to analyze data.
#Count plot of Type.
sns.countplot(x="Type", data=data)


# In[8]:


#Box plot of Type and Weekly_Sales
sns.boxplot(x='Type',y='Weekly_Sales',data=data)


# In[15]:


data["Weekly_Sales"].plot.hist()


# In[13]:


sns.countplot(x="IsHoliday", data=data)


# In[14]:


#Getting the Null values from the data
data.isnull().sum()


# In[16]:


#heatmap to get the Null data
sns.heatmap(data.isnull(),yticklabels=False, cmap="viridis")


# In[43]:


#Deleting the irrelevant data
data1=data.drop(['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'],axis=1)
data1.head(10)


# In[44]:


data1.describe()


# In[18]:


#Converting IsHoliday in Holiday which is integer and 1 for holiday and 0 otherwise.
data['Holiday']=[int(i) for i in list(data.IsHoliday)]
data.head(10)


# In[19]:


Type_dummy=pd.get_dummies(data['Type'],drop_first=True)
Type_dummy.head(10)


# In[20]:


#Concating type_dummy with data
data=pd.concat([data,Type_dummy],axis=1)
data.head(10)


# In[21]:


data=data.drop(['Type','IsHoliday'],axis=1)
data.drop(10)

Now, we perform learning tasks on this data in four steps.
-Splitting the train and test data.
-Applying linear regression.
-Predicting the value
-Evaluate the model
# In[22]:


#Splitting data into train and test data. The size of the test data is 30%.
#splitting data in input and output
X=data.drop(['Weekly_Sales','Store','Date'],axis=1)
y=data['Weekly_Sales']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[35]:


#Applying linear regression and fit the training data into it.
LR=LinearRegression(normalize=True)
LR.fit(X_train,y_train)


# In[31]:


#redicting the data for test value as per linear regression.
y_pred=LR.predict(X_test)
plt.plot(y_test,y_pred,'ro')
plt.plot(y_test,y_test,'g-')
plt.show()


# In[32]:


#Evaluating the model by calculating errors by the root mean square error and R -squared.
Root_mean_square_error=np.sqrt(np.mean(np.square(y_test-y_pred)))
print(Root_mean_square_error)


# # Prediction

# In[36]:


#we give particular tuple to input in the model and predict the weekly sales as output
prediction=LR.predict(pd.DataFrame([(40.37,2.876,173.325456,7.934,103464,0,0,0)]))
print(prediction)


# In[ ]:




