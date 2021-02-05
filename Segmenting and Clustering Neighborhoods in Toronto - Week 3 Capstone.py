#!/usr/bin/env python
# coding: utf-8

# <H1> Segmenting and Clustering Neighborhoods in Toronto Capstone <H1>

# <H2> By: Sian Bhari <H2>

# Installing & Importing the required libraries.

# In[1]:


get_ipython().system('pip install beautifulsoup4')
get_ipython().system('pip install lxml')
get_ipython().system('pip install folium')
get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install matplotlib')
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import matplotlib.colors as colors


# Web-scraping the Wikipedia page for the table of postal codes of Canada; using BeautifulSoup Library of Python

# In[2]:


wiki = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
soup=BeautifulSoup(wiki, 'html.parser')


# In[3]:


from IPython.display import display_html
tab = str(soup.table)
display_html(tab,raw=True)


# The html table is converted to Pandas DataFrame for cleaning and preprocessing.

# In[4]:


dfs = pd.read_html(tab)
df=dfs[0]
df.head(12)


# Dropping the rows where Borough is 'Not assigned'

# In[5]:


df1 = df[df.Borough != 'Not assigned']


# Combining the neighbourhoods with same Postalcode 

# In[6]:


df2 = df1.groupby(['Postal Code','Borough'], sort=False).agg(', '.join)
df2.reset_index(inplace=True)


# Replacing the name of the neighbourhoods which are 'Not assigned' with names of Borough 

# In[7]:


df2['Neighbourhood'] = np.where(df2['Neighbourhood'] == 'Not assigned',df2['Borough'], df2['Neighbourhood'])


# Renaming column names 

# In[8]:


df2.rename(columns = {'Postal Code':'PostalCode'}, inplace = True) 
df2.rename(columns = {'Neighbourhood':'Neighborhood'}, inplace = True)
df2.head(12)


# In[9]:


df2.shape


# Importing the csv file with the latitudes & longitudes for various neighbourhoods in Canada

# In[10]:


lat_lon = pd.read_csv('https://cocl.us/Geospatial_data')
lat_lon.head(12)


# Merging the tables for getting the Latitudes & Longitudes for various neighbourhoods in Canada

# In[11]:


lat_lon.rename(columns={'Postal Code':'PostalCode'},inplace=True)
df3 = pd.merge(df2,lat_lon,on='PostalCode')
df3.head(12)


# Getting all the rows from the data frame which contains Toronto in their Borough.

# In[12]:


df4 = df3[df3['Borough'].str.contains('Toronto',regex=False)]
df4


# Visualizing all the Neighbourhoods of the above data frame using Folium

# In[13]:


import folium
map_t = folium.Map(location=[43.651070,-79.347015],zoom_start=10)

for lat,lng,borough,neighborhood in zip(df4['Latitude'],df4['Longitude'],df4['Borough'],df4['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
    [lat,lng],
    radius=5,
    popup=label,
    color='blue',
    fill=True,
    fill_color='#3186cc',
    fill_opacity=0.7,
    parse_html=False).add_to(map_t)
map_t


# Clustering neighbourhoods

# In[14]:


k=5
toronto_clustering = df4.drop(['PostalCode','Borough','Neighborhood'],1)
kmeans = KMeans(n_clusters = k,random_state=0).fit(toronto_clustering)
kmeans.labels_
df4.insert(0, 'Cluster Labels', kmeans.labels_)


# In[15]:


df4.head(12)


# In[16]:



map_clusters = folium.Map(location=[43.651070,-79.347015],zoom_start=10)

x = np.arange(k)
ys = [i + x + (i*x)**2 for i in range(k)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

markers_colors = []
for lat, lon, neighbourhood, cluster in zip(df4['Latitude'], df4['Longitude'], df4['Neighborhood'], df4['Cluster Labels']):
    label = folium.Popup(' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters

