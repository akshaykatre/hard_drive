
# coding: utf-8

# ## Hard Drive Data
# 
# This is an early exploration on the hard drive 
# dataset 
# 
# 

# In[1]:


## Import the libraries 
import pandas as pd 
import numpy as np 
from matplotlib import pyplot as plt 

get_ipython().magic('matplotlib inline')


# In[2]:


data = pd.read_csv("data/harddrive.csv")


# In[3]:


data.head(10)


# In[4]:


## The number of failures
data.failure.value_counts()


# There are too few failures - only 215 overall. 
# Lets try and see how many failures there are per model 

# In[5]:


models = data.model.unique()
for m in models: 
    try: 
        print("The model, ", m, " has ", data[data["model"]==m].failure.value_counts()[1], " failures")
    except KeyError:
        print("The model, ", m, " has no failures")


# The model ST4000DM000 has the most number of failures. Lets have a look at it closer 

# In[6]:


## Look at all the features
data[data["model"]=="ST4000DM000"].columns


# In[7]:


## Lets look at the SMART features - only the raw ones
smart_features_raw = [x for x in data.columns if "smart" in x and "raw" in x]
print("Number of raw smart features: ", len(smart_features_raw))
smart_features_norm = [x for x in data.columns if "smart" in x and "normalized" in x]
print("Number of normed smart features: ", len(smart_features_norm))


# In[8]:


type(data.smart_199_raw)
len(data.smart_199_raw), data.shape[0]
print(data.smart_199_raw.shape[0])
data.smart_199_raw.name
data.smart_255_raw.isnull().sum()


# In[9]:


def howmanynulls(a, df=None):
    if type(a) != list: 
        print("% of null in ", a.name, " is ", a.isnull().sum()/a.shape[0] * 100) 
    elif type(a) == list:
        for a1 in a:
            howmanynulls(df[a1])
    


# In[10]:


howmanynulls(data.smart_255_raw)
howmanynulls(smart_features_raw, df=data)


# In[11]:


howmanynulls(smart_features_norm, df=data)


# So it turns out some of the features are just null - no matter what type of hard drive it is. 
# 
# In which case, maybe we can get rid of those.
# So the ones to get rid of are: 
# 200, 201, 220, 222, 223, 224,225, 226, 250, 251,252, 254, 255, 
# 195, 22, 11
# 
# 

# In[12]:


for i in [200, 201, 220, 222, 223, 224,225, 226, 250, 251,252, 254, 255, 195, 22, 11]:
    try:
        smart_features_raw.remove('smart_'+str(i)+'_raw')
    except ValueError: 
        print("Index not found: ", i)
    


# In[13]:


smart_features_raw


# In[14]:


def mmm(a, df=None):
    if type(a) != list:
        print("For ", a.name, "mean: ", str(a.mean()), " median: ", str(a.median()), " mode: ", str(a.mode()))
    else:
        for a1 in a:
            mmm(df[a1])


# In[ ]:


data.smart_9_raw

