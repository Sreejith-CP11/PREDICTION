
# coding: utf-8

# In[13]:


import pandas as pd
import matplotlib.pylab as plt


# In[3]:


data=pd.read_csv('http://bit.ly/w-data')


# In[5]:


data.head()


# In[15]:


plt.scatter(data['Hours'],data['Scores'])


# In[7]:


data.columns


# In[11]:


from sklearn.linear_model import LinearRegression as lr
model=lr()
model.fit(data['Hours'].reshape(-1,1),data['Scores'].reshape(-1,1))


# In[16]:


print('If a person studies for 9.25 hrs then the score would be: {0}'.format(model.predict(9.25)))

