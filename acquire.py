#!/usr/bin/env python
# coding: utf-8

# In[12]:


import env
import pandas as pd
import os


# In[20]:


# Make a function named get_titanic_data that returns the titanic data from the codeup data science 
# database as a pandas data frame. Obtain your data from the Codeup Data Science Database. 

def get_connection(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
def get_titanic_data():
    filename = 'titanic.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))
        df.to_csv(filename)
        return df


# In[21]:


# Make a function named get_iris_data that returns the data from the iris_db on the codeup data science database 
# as a pandas data frame. The returned data frame should include the actual name of the species in addition 
# to the species_ids. Obtain your data from the Codeup Data Science Database.

def get_iris_data():
    filename = 'iris.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('SELECT * FROM measurements as m JOIN species AS s ON m.species_id = s.species_id'
                         ,get_connection('iris_db'))
        df.to_csv(filename)
        return df


# In[16]:


# Once you've got your get_titanic_data and get_iris_data functions written, now it's time to add caching to them. 
# To do this, edit the beginning of the function to check for a local filename like titanic.csv or iris.csv. 
# If they exist, use the .csv file. If the file doesn't exist, then produce the SQL and pandas necessary 
# to create a dataframe, then write the dataframe to a .csv file with the appropriate name.


# In[ ]:




