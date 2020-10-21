
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


data=pd.read_csv('Sample - Superstore.csv')


# In[5]:


data.head()


# In[31]:


#plt.figure(figsize=(14,10))
sns.FacetGrid(data,hue='Segment',size=5).map(sns.distplot,'Profit').add_legend()
plt.show()


# In[32]:


sns.FacetGrid(data,hue='Segment',size=5).map(sns.distplot,'Sales').add_legend()
plt.show()


# In[6]:


data.columns


# In[7]:


data['Block_id']=data.groupby(['City','State','Region']).ngroup()


# In[8]:


dd=pd.merge(data,pd.DataFrame(data.groupby('Block_id')['Profit'].apply(np.sum)).reset_index(),on='Block_id',how='inner')


# In[15]:


sns.distplot(data['Profit'])
plt.show()


# ### As we can see some of the datapoints are having negative profit value (i.e is loss), as a anager we need to concentrate on those datapoints and corresponding area

# In[16]:


pd.DataFrame(data.groupby('Block_id')['Profit'].apply(np.sum)).reset_index()['Profit'].plot()
plt.show()


# In[17]:


np.percentile(pd.DataFrame(data.groupby('Block_id')['Profit'].apply(np.sum)).reset_index()['Profit'],23)


# 23 % of areas (Block_id) are suffering from loss

# ##### Weak areas

# In[18]:


temp=pd.DataFrame(data.groupby('Block_id')['Profit'].apply(np.sum)).reset_index()
print('Weak areas:\n')
for i in temp[temp['Profit']< 0]['Block_id'].values:
    print(data[data['Block_id']==i][['City','State','Region']].values[0])


# ### These are the manufacturers that manager needs to concentrate

# In[35]:


data[data['Profit']<0]['Manufacturer'].value_counts()

