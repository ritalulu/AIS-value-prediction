# Databricks notebook source
# MAGIC %md # __Exploratory Data Analysis__

# COMMAND ----------

# MAGIC %md In the following few cells, we load the necessary packages into Python, then load the csv files for both the test & train datasets into their own dataframes.

# COMMAND ----------

import numpy             as np
import pandas            as pd
import matplotlib        as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk

from sklearn import pipeline
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper
from sklearn_pandas import gen_features
import sklearn.preprocessing, sklearn.decomposition, \
       sklearn.linear_model,  sklearn.pipeline, \
       sklearn.metrics

np.__version__, pd.__version__, mpl.__version__, sns.__version__

# COMMAND ----------

ais_train_df = pd.read_csv('/dbfs/mnt/group-ma755/data/ais-train.csv', dtype={'vessel_id': str})
ais_test_df = pd.read_csv('/dbfs/mnt/group-ma755/data/ais-test.csv')


# COMMAND ----------

# MAGIC %md For the next few cells we will take a look at a few of the columns using the .describe command.  

# COMMAND ----------

ais_train_df[['Status']].describe()

# COMMAND ----------

# MAGIC %md The highest occuring status is "Laden".  That makes sense as they ships would spend much more time at sea than in port.  As for "Ballast" status, one reason that might occur less often is if the ship heads to a different, close port to resupply. 

# COMMAND ----------

ais_train_df[['Average']].describe()

# COMMAND ----------

# MAGIC %md The mean 'Average' amount is $11,936.  This is significantly lower than the maximum of $42,709.  Looking at the interquartile ranges above it seems that the $42K is an outlier.

# COMMAND ----------

ais_train_df[['dwt']].describe()

# COMMAND ----------

# MAGIC %md The minimum dead weight tonnage, at 150,400, is very close the to lowest value for Capesize ships (150,000).  The maximum value for Capesize ships exceeds 400,000.  The maximum value in our dataset is only 233,600, so we're dealing a variety of ships in the lower end of the spectrum as far as dead weight tonnage is concerned.

# COMMAND ----------

# MAGIC %md Here we take a look at the data in general, just to get a brief overview.

# COMMAND ----------

ais_train_df

# COMMAND ----------

# MAGIC %md In the next cell, we calculate the average earnings per ship, sorted by the vessel_id.

# COMMAND ----------

ais_train_df[['Average','vessel_id']].groupby('vessel_id',sort=True).mean()

# COMMAND ----------

# MAGIC %md Next we again calculate the average earnings per ship, but this time we calculate the average earnings according to the status as well.

# COMMAND ----------

pd.pivot_table(ais_train_df, 
               values ='Average', 
               index  ='vessel_id', 
               columns='Status', 
               aggfunc=np.mean)

# COMMAND ----------

# MAGIC %md Interestingly, in many cases there seems to be little variation in the averages across the three statuses.  In some cases "Laden" has the lowest average for a ship, and in others it has the highest.

# COMMAND ----------

# MAGIC %md Next we convert the "Date" column from an object into a datetime variable using the Pandas function ".to_datetime".

# COMMAND ----------

ais_train_df['Date']=pd.to_datetime(ais_train_df.Date)

# COMMAND ----------

ais_train_df.sort_values(by=['Date'])

# COMMAND ----------

ais_train_df.sort(['Date'], ascending=False)

# COMMAND ----------

import datetime as dt

# COMMAND ----------

ais_train_df['Date']= ais_train_df['Date'].apply(lambda x:x.date())

# COMMAND ----------

ais_train_df

# COMMAND ----------

# MAGIC %md The following cell shows the average dead weigh tonnage of all the ships operating on a particular day.

# COMMAND ----------

ais_train_df.groupby('Date')['dwt'].mean()

# COMMAND ----------

# MAGIC %md Here we take a look at the number of ships in service on a particular day.

# COMMAND ----------

ais_train_df.groupby('Date')['vessel_id'].count()

# COMMAND ----------

# MAGIC %md The following cell creates a new dataframe based on the ais_train_df dataframe.

# COMMAND ----------

ais_train_df2=ais_train_df.sort_values(by=['Date'])
ais_train_df2['Date']=pd.to_datetime(ais_train_df2.Date)
dates=ais_train_df2['Date']
average=ais_train_df2['Average']
ais_t_df=pd.concat([dates, average], axis=1)
ais_t_df

# COMMAND ----------

# MAGIC %md Next we drop all the duplicate values from the newly created dataframe. 

# COMMAND ----------

unique_t_df=ais_t_df.drop_duplicates()
unique_t_df.set_index('Date',inplace=True)

# COMMAND ----------

unique_t_df[0:]

# COMMAND ----------

# MAGIC %md The next cell is a line graph representing the Average amount over the time frame of our dataset.  We can see that the highest earnings are between the end of 2013 and the first half of 2014.

# COMMAND ----------

unique_t_df.plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Date', fontsize=20)
display()

# COMMAND ----------

# MAGIC %md Next we smooth out the jumps by using a rolling average of the "Average" amount, using a time period of 60 days.

# COMMAND ----------

Average = unique_t_df[['Average']]
Average.rolling(60).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('Date', fontsize=20)
display()

# COMMAND ----------

Average.diff().plot(figsize=(20,10), linewidth=5, fontsize=20);
plt.xlabel('Date', fontsize=20);
display();

# COMMAND ----------

# MAGIC %md Now we will import the autocorrelation_plot function from Pandas, then present the Autocorrelation plot of the "Average" data.

# COMMAND ----------

from pandas.tools.plotting import autocorrelation_plot

# COMMAND ----------

plt.figure()
autocorrelation_plot(Average)
display()

# COMMAND ----------

location_t_df = pd.DataFrame(data= np.c_[ais_train_df['Latitude'], ais_train_df['Longitude'],ais_train_df['Status']],
                     columns= ['Latitude']+['Longitude'] + ['Status'])
location_t_df.head()

# COMMAND ----------

sns.lmplot(x='Longitude', y='Latitude', fit_reg=False, data=location_t_df, col='Status')
display()

# COMMAND ----------

location_date_t_df=pd.DataFrame(data= np.c_[ais_train_df['Date'],ais_train_df['Latitude'], ais_train_df['Longitude'],ais_train_df['Status']],
                     columns= ['Date']+['Latitude']+['Longitude'] + ['Status'])

# COMMAND ----------

location_date_t_df.describe()

# COMMAND ----------

location_date_t_df['Date'] = pd.to_datetime(location_date_t_df['Date']) 

# COMMAND ----------

sns.lmplot(x='Longitude', y='Latitude', fit_reg=False, data=location_date_t_df[location_date_t_df.Date== '2015-12-23'], col='Status')
display()

# COMMAND ----------

sns.lmplot(x='Longitude', y='Latitude', fit_reg=False, data=location_date_t_df[location_date_t_df.Date== '2015-07-21'], col='Status')
display()

# COMMAND ----------

# MAGIC %md __what are we doing in the next cell?__

# COMMAND ----------

a=round(ais_train_df.loc[ais_train_df['Status']== 'Port'],2)
len(a.Latitude.unique())

# COMMAND ----------

# MAGIC %md __FROM HERE__