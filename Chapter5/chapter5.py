
# coding: utf-8

# # Chapter 5 Data Preprocessing, Analysis and Visualization

# In[6]:


import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv( 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv ',sep=';')
df


# In[8]:


array=df.values
array


# In[10]:


x=array[:,0:8]
y=array[:,8]
scaler=MinMaxScaler(feature_range=(0,1))
rescaledX=scaler.fit_transform(x)
np.set_printoptions(precision=3)
rescaledX[0:5,:]


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(x)
rescaledX=scaler.transform(x)
rescaledX[0:5,:]


# In[12]:


from sklearn.preprocessing import Normalizer
scaler=Normalizer().fit(x)
normalizedX=scaler.transform(x)
normalizedX[0:5,:]


# In[13]:


from sklearn.preprocessing import Binarizer
binarizer=Binarizer(threshold=0.0).fit(x)
binaryX=binarizer.transform(x)
binaryX[0:5,:]


# In[15]:


from sklearn.preprocessing import scale
data_standardized=scale(df)
data_standardized.mean(axis=0)


# In[16]:


data_standardized.std(axis=0)


# In[20]:


from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
encoder.fit([[0,1,6,2],[1,5,3,5],[2,4,2,7],[1,0,4,2]])


# In[29]:


from sklearn import preprocessing
encoder = preprocessing.OneHotEncoder()
encoder.fit([  [0, 2, 1, 12], [1, 3, 5, 3],   [2, 3, 2, 12],  [1, 2, 4, 3]])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
encoded_vector


# In[31]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
input_classes=['Havells','Philips','Syska','Eveready','Lloyd']
label_encoder.fit(input_classes)
for i,item in enumerate(label_encoder.classes_):
    print(item,'-->',i)


# In[33]:


df.describe()


# In[34]:


df.shape


# In[35]:


df.head()


# In[37]:


df.tail()


# In[40]:


df.groupby('quality').size()


# In[42]:


df.dtypes


# In[43]:


df.isnull().sum()


# In[44]:


df.describe()


# In[91]:


import matplotlib.pyplot as plt
df.hist(figsize=(20,10))
plt.show()


# In[90]:


df.plot(kind='density',subplots=True,sharex=False,figsize=(10,10))
plt.show()


# In[74]:


df.quality.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))
plt.title("Number of values of each class")
plt.ylabel("Number of values")
plt.xlabel("Class");


# In[101]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
IQR


# In[100]:


df.plot(kind='box',subplots=True,sharex=False,sharey=False,figsize=(22,5))


# In[102]:


plt.figure(figsize=(10,10))
c= df.corr()
plt.imshow(c, cmap='hot', interpolation='nearest')
plt.show()


# In[104]:


import seaborn as sns
plt.figure(figsize=(15,10))
c= df.corr()
sns.heatmap(c,cmap="BrBG",annot=True)
c

