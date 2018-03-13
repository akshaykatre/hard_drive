
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
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
import fuckit
#get_ipython().magic('matplotlib inline')


# In[2]:


data = pd.read_csv("ST4000DM000.csv")


# In[3]:


data.head(10)


# In[4]:


## The number of failures
data.failure.value_counts()


# There are too few failures - only 215 overall. 
# Lets try and see how many failures there are per model 

# In[5]:


''' models = data.model.unique()
for m in models: 
    try: 
        print("The model, ", m, " has ", data[data["model"]==m].failure.value_counts()[1], " failures")
    except KeyError:
        print("The model, ", m, " has no failures")


# The model ST4000DM000 has the most number of failures. Lets have a look at it closer 

# In[6]:


## Look at all the features
data[data["model"]=="ST4000DM000"].columns
 '''

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
        print("For ", a.name, "mean: ", str(a.mean()), " median: ", str(a.median()))
    else:
        for a1 in a:
            mmm(df[a1])


# In[ ]:

## Using the survival prediction models 

time, survival_prob = kaplan_meier_estimator([bool(x) for x in 
    data[data['smart_187_raw']>10]['failure'].values], 
    data[data['smart_187_raw']>10]['smart_9_raw'].values/24.)

plt.step(time, survival_prob, where="post", label='Errors >10')

crit_features = [5, 10, 184, 187, 188, 196, 197, 198, 201]
crit_names = ['smart_'+str(x)+"_raw" for x in crit_features]

mmm(crit_names, df=data)
mmm(smart_features_raw, df=data)

## Narrowing down to the columns that matter! 

smart_features_raw = [x for x in data.columns if "smart" in x and "raw" in x]
## From wiki, most important features
for i in [200, 201, 220, 222, 223, 224,225, 226, 250, 251,252, 254, 255, 195, 22, 11]:
    try:
        smart_features_raw.remove('smart_'+str(i)+'_raw')
    except ValueError: 
        print("Index not found: ", i)

## Find all the columns with a constant or zero values and drop NAs
col_names = []
data = data.dropna(axis=1)
for col in data.columns:
    with fuckit: 
        if max(data[col]) == min(data[col]):
            col_names.append(col)

data = data.drop(col_names, axis=1)

## using only the columns that exist
smart_features_raw = [x for x in smart_features_raw if x in data.columns]

failures = data[data['failure']==1]
success = data[data['failure']==0][data.groupby("serial_number")['failure'].transform('size') < 8]

com = pd.concat([success, failures])

del failures
del success
