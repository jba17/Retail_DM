#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import psycopg2
import sqlalchemy
from sqlalchemy import create_engine


# # Overview
# 
# This assignment here is to understand the dataset provided and ensure data inserted to the table is not null . The dataset is based on a retail domain and comprises of product information , store location , product description and transaction by each unit for a given time period. The end goal is to write the data to a Postgres Database with proper dataset as facts and dimension for reporting purposes.
# 
# 
# ## Course of Action 
# 
# 1. Data Ingestion
# 2. Setting up Connection to Postgres Database
# 3. Understanding each individual dataset provided and data manipulation and cleansing as required.
# 4. Creation of lkp tables,dimensions and facts in the Postgres Database

# ## 1. Data Ingestion
# 

# Directory Path to Dataset

# In[4]:


path ='/Users/jethin/git/Retail_DM/data/'


# In[5]:


df_product = pd.read_csv(path+'product_lookup.csv')
df_fact = pd.read_csv(path+'transactions.csv')
df_store = pd.read_csv(path+'store_lookup.csv')
df_casual = pd.read_csv(path+'causal_lookup.csv')


# ## 2. Connection Setup to Postgres
# 
# 

# In[244]:


#engine = create_engine('postgresql://postgres:JethinAbraham123@13.67.232.170:5432/postgres',echo=False)


# ## 3. Understanding the Dataset Provided

# ### 3.1 Product Dataset

# In[245]:


df_product.head()


# Looking at the Product Size Metric column , it looks like there needs to some data cleansing.
#  --> Ounce ,OZ has to be standardized to OZ
#  --> can convert lb to oz . 1lb =16 oz if needed
#  --> Converting the measure column to integer so that other data with string get converted to Null and can be removed . For eg dataset with ### and %KH

# In[246]:


df_product.product_size.value_counts().head(20)< 10


# In[247]:


df_product['product_size_source'] = df_product.product_size
df_product.product_size=df_product.product_size.str.replace('OUNCE',' OZ',regex =True)
df_product.product_size=df_product.product_size.str.replace('OZ',' OZ',regex =True)
df_product.product_size=df_product.product_size.str.replace('0Z','  OZ',regex =True)
df_product.product_size=df_product.product_size.str.replace('LB',' LB',regex =True)

df_product.product_size=df_product.product_size.str.replace('%','',regex =True)


# Found metric columns starting with a string. Remvoing the records with first character string as the product size should always be an integer and by looking at the dataset it looks correct with certain characters as the first index.

# In[248]:


def first_char(string):
    return string[0]

df_product.product_size = df_product.product_size.apply(lambda x : x.replace(first_char(x),'') 
                                                        if first_char(x).isalpha() else x)

df_product.product_size = df_product.product_size.apply(lambda x :x.strip())

df_product.product_size = df_product.product_size.str.replace(' ','|')


# In[249]:


df_product['measure']=df_product.product_size.str.split('|',1,expand=True)[1]
df_product['measure']=df_product.measure.str.replace('|','').str.strip()
df_product['product_size_new']=df_product.product_size.str.split('|',1,expand=True)[0]


# Created 2 new columns product size new and measure which can be metric columns in the fact later

# In[250]:


df_product.head()


# In[251]:


df_product.product_size_new = pd.to_numeric(df_product.product_size_new, errors='coerce')


# We will be dropping records where the product size column is NA. Based on the below code it would be 37 records
# 

# All the data where the product size starts with a string , blank and '####' are the subset that will be droped

# In[252]:


print(df_product[df_product.product_size_new.isna()].shape)

df_product[df_product.product_size_new.isna()].head(10)


# In[253]:


df_product = df_product.dropna(subset=['product_size_new'])


# In[254]:


df_product =df_product[['upc','product_description','commodity','brand','product_size_new','measure']]


# Converting all categorical columns to lowe case

# In[255]:


df_product.product_description = df_product.product_description.apply(lambda x: x.lower())
df_product.commodity = df_product.commodity.apply(lambda x:x.lower())
df_product.brand =df_product.brand.apply(lambda x:x.lower())


# Checking for Nulls in the subset of data . Looks like there are more rows that have null records

# In[256]:


df_product.isnull().values.any()


# In[257]:


df_product[pd.isnull(df_product).any(axis=1)]


# Looks like the measures were not provided for these records . Since the majority of the dataset was in OZ . We would make a educated guess and set the measure to OZ . But would need the business to intervene . This looks like a legit data and hence adding it to the dimension. If required this could be removed.

# In[258]:


df_product.measure = df_product.measure.apply(lambda x:'OZ' if pd.isnull(x) else x)


# No More Null Records in the Product Dataset

# In[259]:


df_product.isnull().values.any()


# Creating a product dimension from the dataset provided. 

# In[260]:


dim_product=df_product[['brand','product_description','commodity']].drop_duplicates()


# In[261]:


dim_product['product_wid']=dim_product.index.astype(int)

dim_product=dim_product[['product_wid','brand','product_description','commodity']]

dim_product.head()


# ###  3.2 Store Dataset

# From an overview the dataset looks fine. The checks we can do is to find nulls in the dataset and check the length of zipcode assuming all the zipcodes are US zipcodes

# In[262]:


df_store.head()


# Store Data looks good . What we can check is if there are any nulls and the length of the store zip code. Assuming the store zip code is all 5 digits . If otherwise we can treat it as a null as it can be removed or need further clarification.

# In[263]:


df_store[df_store.store_zip_code.astype(str).map(len) !=5]


# Looks like all the zip codes are 5 digits. Lets check for nulls in the dataframe

# In[264]:


df_store.isnull().values.any()


# In[265]:


dim_store=df_store

dim_store['store_wid']=dim_store.index.astype(int)

dim_store =dim_store[['store_wid','store','store_zip_code']]


# ## 3.3 Casual Dataset

# Looking at the dataset the week column looks off. There are only 53 weeks in a year but we have weeks ranging higher than 53. Will take a closer look and make a decision if we would need to remove the dataset.

# In[266]:


df_casual.head()


# In[267]:


df_casual.shape


# No Null Records

# In[268]:


df_casual.isnull().values.any()


# In[269]:


df_casual.feature_desc.value_counts()


# In[270]:


df_casual.display_desc.value_counts()


# In[271]:


df_casual.week.head().value_counts()


# There are only 53 weeks in a day. But we do see many other week numbers like 101 and 102. Since the use case is to remove nulls . We can remove all the records that are higher than 53.

# In[272]:


df_casual = df_casual[df_casual.week <=53]


# In[273]:


dim_casual = df_casual[['feature_desc','display_desc','geography']].drop_duplicates()


# In[274]:


dim_casual['casual_wid']=dim_casual.index.astype(np.int64)


# In[275]:


dim_casual=dim_casual[['casual_wid','geography','feature_desc','display_desc']]


# ##  3.4 Transaction Fact

# In[276]:


df_fact.head()


# In[277]:


df_fact.isnull().values.any()


# In[278]:


df_fact.head(4)


# In[279]:


df_fact.shape


# Joining the Facts and Lkp tables to bring the product size and measure to the fact being metric columns.
# 
# Join as follows
# 
# 
# 1. transaction.upc=product.upc
# 2. transaction.store=store.store
# 3. transaction.upc=casual.upc
#     and transaction.store=casual.store
#     and transaction.week = casual.week
#     and transaction.geography=casual.geography

# In[280]:


df_transaction_fact=df_fact.merge(df_product, how='left',on='upc').merge(df_store,how='left',on='store').merge(df_casual,how='left',on=['upc','store','week','geography'])


# In[281]:


df_transaction_fact.columns


# In[282]:


cols = ['product_wid','store_wid_y','casual_wid','upc','dollar_sales','units','time_of_transaction','geography','week','household','store','basket','day','coupon','product_size_new','measure']


# In[283]:


df_transaction_fact = df_transaction_fact.merge(dim_product, how='left',on=['product_description','commodity','brand']).merge(dim_store,how='left',on='store').merge(dim_casual,how='left',on=['geography','feature_desc','display_desc'])[cols]


# In[284]:


cols = ['product_wid','store_wid_y','casual_wid','upc','dollar_sales','units','time_of_transaction','geography','week','household','store','basket','day','coupon','product_size_new','measure']


# In[285]:


df_transaction_fact.rename(columns={'store_wid_y':'store_wid','product_size_new':'product_measure','measure':'product_unit'},inplace=True)


# The Intention is to build a fact from the transaction dataset . Usually in a real world scenario we would still bring in all the transaction record and place -1 for the casual wid column. Since the excercise is specfically to not add null records to Postgres, droping null records because of casual_wid not having a match would bring down the records to 55909 which is a huge drop. Due to this reason the casual_wid column even though null would be replaced with -1.

# In[286]:


df_transaction_fact.head()


# In[287]:


df_transaction_fact.fillna({'casual_wid':-1}, inplace=True)


# In[289]:


df_transaction_fact.head()


# In[290]:


df_transaction_fact = df_transaction_fact.dropna()


# In[293]:


df_transaction_fact.shape


# ## 4 Creation of Postgress Tables

# In[294]:


#engine = create_engine('postgresql://postgres:JethinAbraham123@13.67.232.170:5432/postgres',echo=False)


# In[295]:


import random
import pandas as pd
from sqlalchemy import create_engine, MetaData
#from postPass import loginDict
dw = 'postgresql://postgres:JethinAbraham123@13.67.232.170:5432/postgres'
dw = create_engine(dw)
from io import StringIO

def to_pg(df, table_name, con):
    data = StringIO()
    
    df.to_csv(data, header=False, index=False)
    data.seek(0)
    raw = con.raw_connection()
    curs = raw.cursor()
    curs.execute("DROP TABLE " + table_name)
    empty_table = pd.io.sql.get_schema(df, table_name, con = con)
    empty_table = empty_table.replace('"', '')
    curs.execute(empty_table)
    curs.copy_from(data, table_name, sep = ',')
    curs.connection.commit()


# In[296]:


get_ipython().magic(u"timeit to_pg(df_product, 'product_lkp', dw)")

get_ipython().magic(u"timeit to_pg(dim_product, 'd_product', dw)")



#df_product.to_sql(name='product_lkp', con=engine, index=False)
#dim_product.to_sql(name='d_product', con=engine, index=False)


# In[297]:


get_ipython().magic(u"timeit to_pg(df_store, 'store_lkp', dw)")

get_ipython().magic(u"timeit to_pg(dim_store, 'd_store', dw)")



#df_store.to_sql(name='store_lkp', con=engine,  index=False)
#dim_store.to_sql(name='d_store', con=engine,  index=False)


# In[298]:


get_ipython().magic(u"timeit to_pg(df_casual, 'casual_lkp', dw)")

get_ipython().magic(u"timeit to_pg(dim_casual, 'd_casual', dw)")


#df_casual.to_sql(name='casual_lkp', con=engine,index=False)
#dim_casual.to_sql(name='d_casual', con=engine,index=False)


# In[305]:


get_ipython().magic(u"timeit to_pg(df_transaction_fact, 'transaction_f', dw)")


# From the data provided below is the data model created having proper Dimension and Fact . Dimension comprise of unique records of Product , Store and Casuals while the transaction fact has the wid columns that join to the dimensions to create a star schema.

# In[303]:


from IPython.display import Image
Image("/Users/jethin/git/retail_dataset/Data_Model.png")


# In[ ]:




