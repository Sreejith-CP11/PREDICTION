
# coding: utf-8

# In[1]:


import sklearn.datasets as datasets
import pandas as pd
import matplotlib.pylab as plt


# In[2]:


data=datasets.load_iris()
X=pd.DataFrame(data.data, columns=data.feature_names)
Y=data.target


# In[3]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X,Y)


# In[19]:


data.target_names


# In[18]:


import graphviz
from sklearn import tree
plt.figure(figsize=(15,8))
dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=data.feature_names,  
                                class_names=data.target_names,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph

