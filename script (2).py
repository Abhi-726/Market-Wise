#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ### DATA LOADING

# In[2]:


exhibitors=pd.read_csv('data/exhibitors.csv')


# In[3]:


visitor=pd.read_csv('data/visitors.csv')


# ### DATA PREPROCESSING

# In[4]:


columns=exhibitors.columns
col_drop=[]
for i ,j in enumerate(exhibitors.isnull().sum()):
    if(j!=0 and j>200):
        col_drop.append(columns[i])
exhibitors.drop(columns=col_drop,inplace=True,axis=1)
col_list=['updatedAt','createdAt','company_repName','company_name','mobile_no','state','city','pro_category','is_confirmed','exhibitor_deliverables_correct']
df=exhibitors
for i in df.columns:
    if i not in col_list:
        df.drop(columns=[i],inplace=True)
df.dropna(axis=0,inplace=True)
df['createdAt']=pd.to_datetime(df['createdAt'])
df['updatedAt']=pd.to_datetime(df['updatedAt'])


# In[5]:


df.info()


# In[6]:


visitor.drop(columns=['vaccine_certificate', 'id_certificate', 'company_logo',
       'alternative_email', 'address_line1', 'address_line2', 'address_line3',
       'company_repProfile', 'firebaseToken', 'socket_id', 'about_me',
       'account_deletion_request','blood_group','designation','visitor','id','email','password','country'],inplace=True)
visitor.dropna(axis=0,inplace=True)
visitor['createdAt']=pd.to_datetime(visitor['createdAt'])
visitor['updatedAt']=pd.to_datetime(visitor['updatedAt'])


# In[7]:


visitor.columns


# In[8]:


x=visitor['profession'].unique()
y=['Engineer Operator looking for new technology',
       'Purchase personnel from boiler manufacturing company',
       'Renewable Energy Project Developer',
       'Purchase Personnel From Boiler Manufacturing Company',
       'Design Engineer Consultant', 'Academician student',
       'Purchase personnel from boiler-user industry',
       'Regulatory Authority Policy Maker Urban Planning Specialist Utility Energy Company',
       'Academicians students engineering colleges',
       'Regulatory authority  policy maker urban planning specialist utility & energy company',
       'Engineer operator looking for new technology', 'R&D Institution',
       'Agencies Dealing In Instrumentation',
       'Agencies dealing in instrumentation',
       'Design engineer consultant',
       'Investor banker ventureCapitalist',
       'Investor banker venture capitalist', 'other']
dicts=dict(zip(x,y))
def clean(x):
    return dicts[x]
visitor['profession']=visitor['profession'].apply(clean)


# In[9]:


x=exhibitors['pro_category'].unique()
y=['Technology serviceprovider', 'Ancillaries',
       'Boiler component manufacturer', 'Boiler manufacturer',
       'Dealers Traders Distributors', 'Turbine manufacturer',
       'Professionals in NDE, energy audit, RLA and R&M',
       'WTP ETP other pollution control equipment manufacturer']
dicts=dict(zip(x,y))
def clean(x):
    return dicts[x]
exhibitors['pro_category']=exhibitors['pro_category'].apply(clean)


# In[10]:


x=visitor[['company_repName', 'company_name','profession']]
y=exhibitors[[ 'company_name','pro_category']]


# In[11]:


y.shape


# In[12]:


df=x.merge(y,how='inner',on=['company_name'])


# In[13]:


df.drop(columns=['company_repName'],inplace=True)


# In[14]:


df['company_name'].nunique()


# ### FEATURE EXTRACTION

# In[15]:


import gensim.downloader as api

# Load a pre-trained Word2Vec model
model = api.load('word2vec-google-news-300')  # Adjust model name and path as needed


# In[16]:


df['profession'].unique()


# ### CALCULATING SIMILARITY

# In[20]:


def sentence_vector(sentence):
  vec = sum([model[word] for word in sentence.split() if word in model.key_to_index]) / len(sentence.split())
  return vec


# In[21]:


rating=[]
for i,profession in enumerate(df['profession']):
    vec1 = sentence_vector(profession)
    vec2 = sentence_vector(df['pro_category'][i])
    # Calculate cosine similarity
    from scipy import spatial
    similarity_score = 1 - spatial.distance.cosine(vec1, vec2)
    print(similarity_score)
    rating.append(similarity_score)


# In[22]:


df['rating']=rating


# In[23]:


df


# In[26]:


def lower(x):
    return x.lower()
df['profession']=df['profession'].apply(lower)


# In[27]:


def correct(x):
    if(x=='purchase personnel from boiler manufacturing company'):
        return 'purchase personnel from boiler-user industry';
    else:
        return x
df['profession']=df['profession'].apply(correct)
df['profession'].unique()


# ### COLLABORATIVE FILTERING

# In[28]:


pivot_table = df.pivot_table(index = ["company_name"],columns = ["profession"],values = "rating")
pivot_table


# In[29]:


pivot_table.fillna(0,inplace=True)


# In[30]:


pivot_table


# ### COSINE SIMILARITY

# In[31]:


from sklearn.metrics.pairwise import cosine_similarity
similarity_score=cosine_similarity(pivot_table)
def recommend(exhibitor):
    index=np.where(pivot_table.index==exhibitor)[0][0]
    similar_items=sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[2:6]
    for i in similar_items:
        print(pivot_table.index[i[0]])


# ### OUTPUT

# In[32]:


recommend('Yaskawa india pvt ltd')


# In[33]:


recommend('Moglix')


# ### EXPORTING MODEL

# In[34]:


pivot_table.to_csv('Pivot_Table.csv')


# In[35]:


import pickle


# In[41]:


pickle_out=open("Ps4.pkl","wb")
pickle.dump([similarity_score,pivot_table],pickle_out)
pickle_out.close()


# In[ ]:




