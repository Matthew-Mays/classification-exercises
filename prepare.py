#!/usr/bin/env python
# coding: utf-8

# ### The end product of this exercise should be the specified functions in a python script named prepare.py. Do these in your classification_exercises.ipynb first, then transfer to the prepare.py file.
# 
# ### Using the Iris Data:
# 
# 1. Use the function defined in acquire.py to load the iris data.

# In[30]:


import acquire
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

import warnings
warning.filterwarnings('ignore')

import numpy as np


# In[2]:


iris_df = acquire.get_iris_data()
iris_df


# 2. Drop the species_id and measurement_id columns.

# In[3]:


iris_df = iris_df.drop(columns=['measurement_id', 'species_id'])
iris_df


# 3. Rename the species_name column to just species.

# In[4]:


iris_df = iris_df.rename(columns={'species_name': 'species'})
iris_df


# 4. Create dummy variables of the species name.

# In[5]:


species_dummies = pd.get_dummies(iris_df.species, drop_first=True)
species_dummies.head()


# In[6]:


iris_df = pd.concat([iris_df, species_dummies], axis=1)
iris_df.tail()


# 5. Create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.

# In[7]:


def prep_iris(cached=True):
    df = acquire.get_iris_data(cached)
    df = df.drop(columns=['Unnamed: 0', 'species_id.1','measurement_id', 'species_id'])
    df = df.rename(columns={'species_name': 'species'})
    species_dummies = pd.get_dummies(df.species, drop_first=True)
    df = pd.concat([df, species_dummies], axis=1)
    
    return df


# In[8]:


iris = prep_iris()
iris.sample(7)


# In[22]:


titanic = acquire.get_titanic_data()
titanic.head()


# In[23]:


titanic = titanic.drop(columns='Unnamed: 0')
titanic.head()


# In[24]:


titanic[titanic.embarked.isnull()]


# In[25]:


titanic[titanic.embark_town.isnull()]


# In[26]:


titanic = titanic[~titanic.embarked.isnull()]
titanic.info()


# In[27]:


titanic = titanic.drop(columns='deck')
titanic.info()


# In[28]:


titanic_dummies = pd.get_dummies(titanic.embarked, drop_first=True)
titanic_dummies.sample(10)


# In[29]:


titanic = pd.concat([titanic, titanic_dummies], axis=1)
titanic.head()


# In[32]:


train_validate, test = train_test_split(titanic, test_size=.2, random_state=123, stratify=titanic.survived)


# In[33]:


train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.survived)


# In[34]:


print(f'train: {train.shape}')
print(f'validate: {validate.shape}')
print(f'test: {test.shape}')


# In[35]:


def titanic_split(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.survived)
    return train, validate, test


# In[36]:


imputer = SimpleImputer(strategy = 'mean')


# In[38]:


train['age'] = imputer.fit_transform(train[['age']])


# In[39]:


train['age'].isnull().sum()


# In[51]:


def impute_mean_age(train, validate, test):
    imputer = SimpleImputer(strategy = 'mean')
    train['age'] = imputer.fit_transform(train[['age']])
    validate['age'] = imputer.fit_transform(validate[['age']])
    test['age'] = imputer.fit_transform(test[['age']])
    
    return train, validate, test


# In[54]:


def prep_titanic(cahced=True):
    
    df = acquire.get_titanic_data()
    df = df[~df.embarked.isnull()]
    titanic_dummies = pd.get_dummies(df.embarked, drop_first=True)
    df = pd.concat([df, titanic_dummies], axis=1)
    df.drop(columns='deck')
    train, validate, test = titanic_split(df)
    train, validate, test = impute_mean_age(train, validate, test)
    
    return train, validate, test


# In[55]:


train, validate, test = prep_titanic()


# In[57]:


print(f'train: {train.shape}')
print(f'validate: {validate.shape}')
print(f'test: {test.shape}')

