
# coding: utf-8

# # Training and Testing Data

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


df =pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv')


# In[4]:


df.head()


# In[7]:


y=df.temp
x=df.drop('temp',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.head()


# In[8]:


x_train.shape


# In[9]:


y_train.shape


# In[10]:


x_test.head()


# In[11]:


x_test.shape


# In[19]:


from sklearn.datasets import load_iris
iris=load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.5,test_size=0.5)
y_test


# In[20]:


y_train

