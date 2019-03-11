#!/usr/bin/env python
# coding: utf-8

# ## Readmission Dataset: Exploratory Data Analysis¶
#     Here, I will be analysing the Readmission dataset. This is a supervised classification dataset. 
#     Variables could be categorical or numerical. There are different statistical and visualization techniques of investigation for each type of variable.
#     Below are the questions that i will be addressing via exploratory analysis, which will give us more insight into the dataset:
#     What age range,gender, were most readmitted?
#     Does Weight contribute to readmission?
#     what factors helped reduce Readmission?
#     How does time in hospital relate to readmission?

# In[40]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import warnings
warnings.filterwarnings("ignore")


# In[41]:


# Loading Dataset
readmission=pd.read_csv("diabetic_data.csv")


# In[42]:


#I observed some variables, they don't hold much information in our case,removing them will not affect our analysis
readmission=readmission.drop(columns=[ 'admission_type_id','discharge_disposition_id','admission_source_id'
                           ,'encounter_id','patient_nbr'])


# In[43]:


#peeping into readmission dataset
readmission.head(n=4)


# In[45]:


readmission=readmission.replace('?',np.nan)


# In[46]:


#peeping into readmission dataset
readmission.head(n=4)


# In[47]:


# checking the datatypes available again
readmission.dtypes.head(n=6)


# In[98]:


readmission.describe()


# In[97]:


ax = sns.boxplot(readmission["time_in_hospital"])


# The table above describe all numerical variables.; Comparing mean and 50% percentile will give hint if data are skewed or not.

# ## What age range,gender, were most readmitted?
# >   Let's look at some personal information about the passengers as related.

# In[112]:


readmitted=readmission[readmission.readmitted=='<30']
readmission_age_class=readmitted[['age','readmitted']].groupby(['age']).count().reset_index()
readmission_age_class


# In[113]:


sns.barplot(x='age',y='readmitted',data=readmission_age_class)
sns.despine(offset=10, trim=True)
plt.show()


# >   The above table and graph shows how readmission increases as age increases,the graph also shows that more readmissions occurs between age 60 and 90.
# 
# >   Let's compare age frequencies accross all categories of readdmission variable, this will give us hint about relationship between age and readmission.
# 

# In[114]:


# create a countplot
#Categorical vs Categorical
sns.countplot(x='age',data=readmission,hue = 'readmitted')
# Remove the top and down margin
sns.despine(offset=10, trim=True)
# display the plot
plt.show()


# >  Above new graph shows more insight, overall the frequency of 'NO' readmission is far more than  >30 & <30  days 
#     readmission frequencies.
#     
# >  Nevertheless, the graph shows that patients of age range 50 and 90 visits hospital/clinic more than other age ranges 
# 

# In[159]:


f,axarr=plt.subplots(2,1,figsize=(11,22))
sns.countplot(x='gender',data=readmission,ax=axarr[0])
sns.countplot(x='gender',data=readmission,hue = 'readmitted',ax=axarr[1])
sns.despine(offset=10, trim=True)
plt.show()


# >  Generally, more numbers of female appeared to be readmitted than men, but when we calculate percentage of the gender readmissions, the female readmissions is not much higher than that of men, as shown in the figures below.    
# 

# In[209]:


readmission_gender_class = readmission.groupby(['gender','readmitted']).agg({'readmitted': 'count'}).rename(columns={'readmitted':'readmitted_count_pct'})#.reset_index()
readmission_gender_class_pcts = readmission_gender_class.groupby(level=0).apply(lambda x:
                                                 100 * x / float(x.sum())).reset_index()
print(readmission_gender_class_pcts)

sns.barplot(x='gender',y='readmitted_count_pct',data=readmission_gender_class_pcts,hue='readmitted')
sns.despine(offset=10, trim=True)
plt.show()


# ## Does Weight contribute to readmission?
# >  By the way weight is already converted to categorical data type via binning, so i will be investigating how weight affect readmission
#     
#      

# In[247]:


#Bivariate  analysis
#weight (categorical) vs readmitted (categorical)
from scipy.stats import chi2_contingency
contingency = pd.crosstab(readmission['weight'],readmission['readmitted'])
contingency.sort_values(by=['<30'], ascending=False)


# >  Based on the dataset,number of <30 readmission are high for weight range from 50 to 150.
