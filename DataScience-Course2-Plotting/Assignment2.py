
# coding: utf-8

# # Assignment 2
# 
# Before working on this assignment please read these instructions fully. In the submission area, you will notice that you can click the link to **Preview the Grading** for each step of the assignment. This is the criteria that will be used for peer grading. Please familiarize yourself with the criteria before beginning the assignment.
# 
# An NOAA dataset has been stored in the file `data/C2A2_data/BinnedCsvs_d25/391a2922ad597ba080f4b99dea6d62842562d64845ef5df1a384561e.csv`. The data for this assignment comes from a subset of The National Centers for Environmental Information (NCEI) [Daily Global Historical Climatology Network](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt) (GHCN-Daily). The GHCN-Daily is comprised of daily climate records from thousands of land surface stations across the globe.
# 
# Each row in the assignment datafile corresponds to a single observation.
# 
# The following variables are provided to you:
# 
# * **id** : station identification code
# * **date** : date in YYYY-MM-DD format (e.g. 2012-01-24 = January 24, 2012)
# * **element** : indicator of element type
#     * TMAX : Maximum temperature (tenths of degrees C)
#     * TMIN : Minimum temperature (tenths of degrees C)
# * **value** : data value for element (tenths of degrees C)
# 
# For this assignment, you must:
# 
# 1. Read the documentation and familiarize yourself with the dataset, then write some python code which returns a line graph of the record high and record low temperatures by day of the year over the period 2005-2014. The area between the record high and record low temperatures for each day should be shaded.
# 2. Overlay a scatter of the 2015 data for any points (highs and lows) for which the ten year record (2005-2014) record high or record low was broken in 2015.
# 3. Watch out for leap days (i.e. February 29th), it is reasonable to remove these points from the dataset for the purpose of this visualization.
# 4. Make the visual nice! Leverage principles from the first module in this course when developing your solution. Consider issues such as legends, labels, and chart junk.
# 
# The data you have been given is near **New Delhi, National Capital Territory of Delhi, India**, and the stations the data comes from are shown on the map below.

# In[495]:

get_ipython().magic('matplotlib notebook')


# In[496]:

import matplotlib.pyplot as plt
import mplleaflet
import pandas as pd

def leaflet_plot_stations(binsize, hashid):

    df = pd.read_csv('data/C2A2_data/BinSize_d{}.csv'.format(binsize))

    station_locations_by_hash = df[df['hash'] == hashid]

    lons = station_locations_by_hash['LONGITUDE'].tolist()
    lats = station_locations_by_hash['LATITUDE'].tolist()

    plt.figure(figsize=(8,8))

    plt.scatter(lons, lats, c='r', alpha=0.7, s=200)

    return mplleaflet.display()

leaflet_plot_stations(25,'391a2922ad597ba080f4b99dea6d62842562d64845ef5df1a384561e')


# In[497]:

df = pd.read_csv('data/C2A2_data/BinnedCsvs_d100/4e86d2106d0566c6ad9843d882e72791333b08be3d647dcae4f4b110.csv')
df.sort_values("Date" ,inplace=True)
min_df = df[df["Element"] == "TMIN"] 
max_df = df[df["Element"] == "TMAX"] 
min_df15 = df[(df['Element'] == 'TMIN') & (df.Date.str[0:4]  == '2015')]
max_df15 = df[(df['Element'] == 'TMAX') & (df.Date.str[0:4]  == '2015')]

df.tail()
min_df15.set_index("Date", inplace=True)
min_df15 = min_df15["Data_Value"]
max_df15.set_index("Date", inplace=True)
max_df15 = max_df15["Data_Value"]


# In[498]:

import numpy as np
min_df = min_df.groupby(["Date"])["Data_Value"].min()
max_df = max_df.groupby(["Date"])["Data_Value"].max()


# In[499]:

max_df15.head()


# In[500]:

time_range = pd.date_range('2005-01-01', '2015-12-31')
min_df.index = pd.DatetimeIndex(min_df.index)
max_df.index = pd.DatetimeIndex(max_df.index)
min_df = min_df.reindex(time_range, method='ffill')
max_df = max_df.reindex(time_range, method='ffill')


# In[501]:

for i in min_df.index:
    
    #print(i.month, i.day)
    if (i.month == 2) and (i.day == 29):
        min_df.drop(i, inplace=True)

for i in max_df.index:
    #print(i.month, i.day)
    if (i.month == 2) and (i.day == 29):
        max_df.drop(i, inplace=True)


# In[502]:

i1 = min_df.index.tolist()
i1 = list(map(pd.to_datetime, i1))


# In[503]:

print(len(min_df), len(max_df))


# In[504]:

i2 = max_df.index.tolist()
i2 = list(map(pd.to_datetime, i2))


# In[505]:

print(len(i1), len(i2))


# In[506]:

i1_15 = min_df15.index.tolist()
i1_15 = list(map(pd.to_datetime, i1_15))


# In[507]:

i2_15 = max_df15.index.tolist()
i2_15 = list(map(pd.to_datetime, i2_15))


# In[508]:

min_df.index.month[0], min_df.index.day[0]


# In[509]:

# Now to check conditional on 2015 data, I'll collate the Record highs and Record lows from 2005 - 2014 in two series
month_day = set(str(i) + "-" + str(j) for i, j in zip(min_df.index.month, min_df.index.day))
month_day = list(month_day)
decade_high = {}
decade_low = {}

for element in month_day:
    elem = "2015-" + element
    greatest = 0
    for i, j, v in zip(max_df.index.month, max_df.index.day, max_df):
        if element == str(i) + "-" + str(j):
            if v > greatest:
                greatest = v
            
    decade_high[elem] = greatest
    

for element in month_day:
    elem = "2015-" + element
    lowest = 99999
    for i, j, v in zip(min_df.index.month, min_df.index.day, min_df):
        if element == str(i) + "-" + str(j):
            if v < lowest:
                lowest = v
            
    decade_low[elem] = lowest
################################################################
#decade_low


# In[510]:

decade_high = pd.DataFrame({"Date": pd.Series(decade_high).index, "Value": pd.Series(decade_high).values})
decade_low = pd.DataFrame({"Date": pd.Series(decade_low).index, "Value": pd.Series(decade_low).values})


# In[511]:

decade_high["Date"] = pd.to_datetime(decade_high["Date"])
decade_low["Date"] = pd.to_datetime(decade_low["Date"])


# In[512]:

decade_high.sort_values("Date", inplace=True)
decade_low.sort_values("Date", inplace=True)


# In[513]:

decade_low.head()


# In[514]:

decade_high.set_index("Date", inplace=True)

decade_low.set_index("Date", inplace=True)


# In[515]:

decade_low.head()


# In[516]:

# Now, checking for the 2015 conditional in the next few cells
# We will compare min_df15 with decade_low and max_df15 with decade_high 
# So first, we need them to have the same index vals and the same length
# Let's do that now
min_df15.head()


# In[517]:

min_df15 = min_df15.groupby(min_df15.index).min()
max_df15 = max_df15.groupby(max_df15.index).max()
#Now, the lengths are equal. Both equal 365 now, that is the number of days in 2015
#overlay_low = [min_df15[min_df15[i] < decade_low[i]] for i in range(len(min_df15))]


# In[518]:

#decade_low = decade_low.reset_index
#type(decade_low)#[1]

decade_low = decade_low.T.iloc[0] # Converting DataFrame to Series to make it eligible for comparing
decade_high = decade_high.T.iloc[0] # Converting DataFrame to Series to make it eligible for comparing
#[print(min_df15[i] , decade_low[i]) for i in range(len(min_df15))]
#broken_min = np.where(temp_min_15['Data_Value'] < temp_min['Data_Value'])[0]


# In[519]:

#overlay_low = [(min_df15[i] < decade_low[i]) for i in range(len(min_df15))]
overlay_low = np.where(min_df15 <= decade_low)
overlay_high = np.where(max_df15 >= decade_high)

overlay_high = max_df15.iloc[overlay_high]
overlay_low = min_df15.iloc[overlaw_low]
overlay_high.head()


# In[520]:

plt.figure()
plt.plot(i1, min_df, "-", c='b', label="Record Low")
plt.plot(i2, max_df, "-", c='r', label="Record High")
#plt.scatter(i1_15, min_df15, c="k", label="2015low")
plt.scatter(overlay_high.index, overlay_high, c="k", label="Decade Record Breaker [2015] (High)", s=10)
plt.scatter(overlay_low.index, overlay_low, c="g", label="Decade Record Breaker [2015] (Low)", s=10)
plt.gca().fill_between(i1, min_df, max_df, facecolor='yellow', alpha=0.5)
plt.gca().set_ylim([-150, 650])
plt.legend()
plt.xlabel('Day of the Year')
plt.ylabel('Temperature (Tenths of Degrees Celsius)')
plt.title('Temperature Summary Plot near New Delhi, India')
plt.xticks(rotation = '45')
plt.subplots_adjust(bottom=0.2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()

