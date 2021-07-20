#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:54:32 2020

@author: adieleejiofor
"""
# INPUT TENDER Rounds Here
x = [124,125,126]
y = [124,125,126]
a = 123
b = 124
c = 125
d = 126


import pandas as pd
from matplotlib import pyplot as plt
from datetime import date, timedelta
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
from dateutil.relativedelta import *
import plotly.express as px
from scipy import stats
from scipy.stats import kurtosis, skew
import plotly
from plotly import __version__
print(__version__)
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

#pd.options.mode.chained_assignment = None

def dynamic_accepted(x = None):
    data = pd.read_excel('Trial.xlsx')
    if x is not None:
        data = data.loc[data['Tender Round'].isin(x)]
    data = data[['Tender Date','Start Date', 'End Date', 'Reasoning','Tender Round', 'Tender Ref','Company Name','Tendered Unit','Status','Weekdays (From)', 'Weekdays (To)', 'Start EFA', 'End EFA','Static/Dynamic','Availability Fee (£/h)','Dynamic Primary (0.5Hz)', 'Dynamic Secondary (0.5Hz)', 'Dynamic High (0.5Hz)', 'Duration (h)']]
    data = data[data["Static/Dynamic"] == 'Dynamic']
    #data = data[data['Status'] == 'Rejected']
    data = data[data['Reasoning'] != 'Multiple tenders received' ]
    data = data[data['Reasoning'] != 'Not meeting prerequisites' ]
    duplicates = data[data.duplicated(['Tender Round','Tendered Unit', 'Start EFA', 'End EFA', 'Status'], keep = False)]
    test = duplicates.sort_values(['Tender Round', 'Tendered Unit', 'Start EFA'], ascending = [False, True, True])
    def func(row):
        if row['Dynamic High (0.5Hz)'] == 0:
            return row['Dynamic Primary (0.5Hz)']
        return None
    test['Dynamic High (0.5Hz)'] = test.apply(func, axis=1)
    test = test.dropna()
    for i in duplicates.index:
        data.drop(i, inplace=True)
    frames = [data, test]
    result = pd.concat(frames)
    #result_filtered = result[(result['Dynamic Primary (0.5Hz)'] != result['Dynamic High (0.5Hz)']) & (result['Dynamic Secondary (0.5Hz)'] != result['Dynamic High (0.5Hz)'])]
    result['price'] = np.where(((result['Dynamic Primary (0.5Hz)']==0) | (result['Dynamic Secondary (0.5Hz)']==0)),
                               result['Availability Fee (£/h)']/result[['Dynamic Primary (0.5Hz)', 'Dynamic Secondary (0.5Hz)','Dynamic High (0.5Hz)']].max(axis=1),
                               result['Availability Fee (£/h)']/result[['Dynamic Primary (0.5Hz)', 'Dynamic Secondary (0.5Hz)','Dynamic High (0.5Hz)']].mean(axis=1))
    result = result[result['price'] != 0]
    result['Time_dif'] = result['End Date'] - result['Tender Date']
    result['Time_dif'] = pd.to_numeric(result['Time_dif'].dt.days, downcast = 'integer')
    result = result[result['Time_dif'] <= 62]
    return result


#maxi = result['Tender Round'].max()

result = dynamic_accepted(x)

result['Tender Round'].max()

df19 = result
df19['Start EFA'] = (pd.to_numeric(df19['Start EFA'], errors='coerce')).fillna(0).astype(int)
df19['End EFA'] = (pd.to_numeric(df19['End EFA'], errors='coerce')).fillna(0).astype(int)
df19['blocks_nr']='test'
df19['blocks']='test'

for i in range(0, len(df19)):
    if  (df19['End EFA'].iloc[i]-df19['Start EFA'].iloc[i]==1): 
        df19['blocks'].iloc[i] =(df19['Start EFA'].iloc[i])
        df19['blocks_nr'].iloc[i]=1
        
    elif  (df19['End EFA'].iloc[i]-df19['Start EFA'].iloc[i]==0): 
        df19['blocks'].iloc[i] =(range(1,7))
        df19['blocks_nr'].iloc[i]= len(df19['blocks'].iloc[i])
        
    elif (df19['Start EFA'].iloc[i] > df19['End EFA'].iloc[i]):
        df19['blocks'].iloc[i]= range(df19['Start EFA'].iloc[i], 7)
        df19['blocks_nr'].iloc[i]=len(df19['blocks'].iloc[i])
    else:
        df19['blocks'].iloc[i]= range(df19['Start EFA'].iloc[i], df19['End EFA'].iloc[i])
        df19['blocks_nr'].iloc[i]=len(df19['blocks'].iloc[i])

df19 = df19.reindex(df19.index.repeat(df19.blocks_nr)).reset_index()
df19['EFA_block'] = 'adi'

j=0
for i in range(0,len(df19)):
    #print ('==================', i)
    if (df19['blocks_nr'].loc[i]==1):
        #print('mpikame sto prwto if')
        #print('First IF condition\n')
        j=0
        #print(df19['Start EFA'].loc[i], df19['End EFA'].loc[i])
        df19['EFA_block'].loc[i] = 'EFA {}'.format(df19['Start EFA'].loc[i])
        #print('>>>>>>>>>>>>>>> ',df19['EFA_block'].loc[i])
        #print('>>>>>>>>>>>>>>> our EFA block is: ',df19['EFA_block'].loc[i])
    else:
#         print('mpikame sto', df19['blocks'].loc[i], ' to opoio exei length ', len(df19['blocks'].loc[i]))
#         print('poso einai to j? ', j)
        #print('ELSE we are in ', df19['blocks'].loc[i], ' and its length is ', len(df19['blocks'].loc[i]))
        #print('What is j: ', j)
        if(j<len(df19['blocks'].loc[i])):
#             print('to j einai: ',j)
#             print('j is: ', j)
#             print(len(df19['blocks'].loc[i]),len(df19['blocks'].loc[i]))
           # print('element of the range: ', df19['blocks'].loc[i][j])
           
            df19['EFA_block'].loc[i] = 'EFA {}'.format(df19['blocks'].loc[i][j])
#             print('----------------------',df19['EFA_block'].loc[i])

            j+=1
            
        elif(j==len(df19['blocks'].loc[i])):
            j=0
            #print('!!!!!!!!!: ', df19['blocks'].loc[i][j])
            df19['EFA_block'].loc[i] = 'EFA {}'.format(df19['blocks'].loc[i][j])
#             print('----------------------',df19['EFA_block'].loc[i])
            j+=1
        
        elif(j>len(df19['blocks'].loc[i])):
            j=0
           # print('????????: ', df19['blocks'].loc[i][j])
            df19['EFA_block'].loc[i] = 'EFA {}'.format(df19['blocks'].loc[i][j])
#             print('----------------------',df19['EFA_block'].loc[i])
            j+=1
        else:
            j=0


## Price Range

#### Earlier Tender Round

df19_A = df19[df19['Tender Round'] == 124]

df19_grouped2 = df19[df19['Status']== 'Accepted']

df19_grouped3 = df19[df19['Status']== 'Rejected']
df19_120 = df19[df19['Tender Round'] == a]
df19_121 = df19[df19['Tender Round'] == b]
df19_122 = df19[df19['Tender Round'] == c]
df19_123 = df19[df19['Tender Round'] == d]

#df19_A

df19_A = df19_A.sort_values(['Status', 'EFA_block'], ascending = ['True', 'True'] )

#df19_A_grouped = df19_A.groupby('Tender Round','EFA_block').agg(Price = ('price', 'mean'), Volume = ('Dynamic Secondary (0.5Hz', 'sum')).reset_index()
df19_A['Tender Round'].max()
Tender = 'Tender Round ' + str(int(df19_A['Tender Round'].max()))
Title = Tender + ' Accepted & Rejected Dynamic Price Ranges'
fig = px.box(df19_A, x="EFA_block", y = "price", color = 'Status')
fig.update_layout(title = Title, xaxis_title = Tender)
fig.layout.template = 'plotly_dark'
fig.show()
#plot(fig)
#fig.write_image("image.jpeg")

#### Latest TR

df19_B = df19[df19['Tender Round'] == 126]

df19_grouped2 = df19[df19['Status']== 'Accepted']

df19_grouped3 = df19[df19['Status']== 'Rejected']

df19_B = df19_B.sort_values(['Tender Round','EFA_block', 'Status'], ascending = ['True','True', 'True'])

Tender = 'Tender Round ' + str(int(df19_B['Tender Round'].max()))
Title = Tender + ' Accepted & Rejected Dynamic Price Ranges'
fig = px.box(df19_B, x="EFA_block", y = "price", color = 'Status')
fig.update_layout(title = Title, xaxis_title = Tender)
fig.layout.template = 'plotly_dark'
fig.show()
#plot(fig)
#fig.write_image("image.jpeg")

## Experiment

Month = df19["End Date"].max().strftime('%B')
year = (datetime.now() + timedelta(days=0) ).strftime('%Y')
#newpath = os.chidir("_Sent_availability_files_to_client\ " +year)
newpath = r'//Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/Monthly Meeting/' +year + '/' +Month+ '/DFFR/'

if not os.path.exists(newpath):
    os.makedirs(newpath) 
os.chdir(newpath)
#change working directory to save output file

#dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file

current_directory = os.getcwd()

#Edit the excel file
#mypath = current_directory + "/fafr.xlsx"



df19_grouped = df19.groupby(['Tender Round','EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()

# Trends

df19['Date'] = df19['End Date'].dt.strftime('%b-%y')

# df19

df19 = df19.sort_values(['Tender Round','EFA_block', 'Status'], ascending = ['True','True', 'True'])

df19 = df19.sort_values(['EFA_block', 'Status'], ascending = ['True', 'True'])
fig = px.box(df19, x = 'EFA_block',y="price", facet_col="Date", color="Status",
             boxmode="overlay",hover_name = 'Status', points = "all")
title = 'Dynamic FFR Monthly Results'
fig.update_layout(title = title)
fig.layout.template = 'seaborn'
fig.show()
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")


#plot(fig)

# df19

## Market Results

df19_grouped = df19.groupby(['Tender Round','EFA_block', 'Status', 'Date']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()

# df19_grouped

Tender = 'Tender Round ' + str(int(df19['Tender Round'].max()))
Title = Tender + ' Dynamic Market Results & Volumes'
fig = px.scatter(df19_grouped, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Status")
fig.update_layout(title = Title, xaxis_title = Tender)

#fig.show()
#plot(fig)

fig = px.scatter(df19_grouped, x = 'EFA_block',y="avg_price",size = "total_volume",hover_name = 'Status', facet_col="Date", color="Status")
title = 'Dynamic FFR Monthly Volumes'
fig.update_layout(title = title, yaxis_title = 'Price (£/MW)')
#fig.add_scatter(x = dfk_grouped['EFA_block'], y = dfk_grouped['avg_price'])
fig.layout.template = 'seaborn'
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")
fig.show()

#plot(fig)

df19_Ac = df19[df19['Status'] == 'Accepted']

df19_grouped2 = df19_Ac.groupby(['Company Name','Tender Round','EFA_block', 'Status', 'Date']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()

df19_grouped2 = df19_grouped2.sort_values(by='Date')
months = ["Jan-20", "Feb-20", "Mar-20","Apr-20", "May-20","Jun-20","Jul-20","Aug-20","Sep-20","Oct-20","Nov-20","Dec-20"]
# months1 = ["Dec-19", "Jan-20", "Feb-20", "Mar-20",
#            "Apr-20", "May-20", "Jun-20"]
df19_grouped2['Date'] = pd.Categorical(df19_grouped2['Date'], 
                                  categories=months, ordered=True)
df19_grouped2 = df19_grouped2.sort_values(["Date", "EFA_block"])

fig = px.scatter(df19_grouped2, x = 'EFA_block',y="avg_price",size = "total_volume",hover_name = 'Status', facet_col="Date", color="Company Name")
title = 'Accepted Dynamic FFR Monthly Volumes Company '
fig.update_layout(title = title, yaxis_title = 'Price (£/MW)')
#fig.add_scatter(x = dfk_grouped['EFA_block'], y = dfk_grouped['avg_price'])
fig.layout.template = 'seaborn'
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")
fig.show()
# plot(fig)

df19_Rj = df19[df19['Status'] == 'Rejected']
df19_grouped3 = df19_Rj.groupby(['Company Name','Tender Round','EFA_block', 'Status', 'Date']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()
df19_grouped3.sort_values(['Date'], inplace = True)

df19_grouped3 = df19_grouped3.sort_values(by='Date')
months = ["Jan-20", "Feb-20", "Mar-20","Apr-20", "May-20","Jun-20","Jul-20","Aug-20","Sep-20","Oct-20","Nov-20","Dec-20"]
# months1 = ["Dec-19", "Jan-20", "Feb-20", "Mar-20",
#            "Apr-20", "May-20", "Jun-20"]
df19_grouped3['Date'] = pd.Categorical(df19_grouped3['Date'], 
                                  categories=months, ordered=True)
df19_grouped3 = df19_grouped3.sort_values(["Date", "EFA_block"])
fig = px.scatter(df19_grouped3, x = 'EFA_block',y="avg_price",size = "total_volume",hover_name = 'Status', facet_col='Date', color="Company Name")
title = 'Rejected Dynamic FFR Monthly Volumes Company'
fig.update_layout(title = title, yaxis_title = 'Price (£/MW)')
#fig.add_scatter(x = dfk_grouped['EFA_block'], y = dfk_grouped['avg_price'])
fig.layout.template = 'seaborn'
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")
fig.show()
# plot(fig)

## Company Performance
df19_grouped_u = df19.groupby(['Date','Company Name','Tendered Unit','Status']).agg(avg_price=('price', 'mean'), total_volume=('Dynamic Secondary (0.5Hz)', 'mean' )).reset_index()
df19_grouped_u_a = df19_grouped_u[df19_grouped_u['Status'] == 'Accepted']

df19_grouped_u_b = df19_grouped_u_a.groupby(['Date','Company Name']).agg(avg_price=('avg_price', 'mean'), total_volume=('total_volume', 'sum' )).reset_index()

Title = Tender + ' Monthly Company DFFR Market Performance'
fig = px.scatter(df19_grouped_u_b, x = 'total_volume',y="avg_price",size = "total_volume",hover_name = 'Company Name', facet_col='Date', color="Company Name")
#fig = px.scatter(df19_grouped_u_b, x ="total_volume", y ="avg_price", size = "total_volume", color= "Company Name")
fig.update_layout(title = Title, yaxis_title = 'Average Price (£/MWh)')
fig.layout.template = 'plotly_dark'
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + ".png")
fig.show()

## Kiwi Bids

dfk = df19[df19['Company Name'] == 'Kiwi Power Ltd']

dfk_grouped = dfk.groupby(['Tender Round','EFA_block', 'Status', 'Date']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()

Title = 'Kiwi Power DFFR Results'
fig = px.scatter(dfk_grouped, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Status", hover_name = 'Status', facet_col = 'Date')
fig.update_layout(title = Title, xaxis_title = Tender)
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + ".png")
fig.show()
#plot(fig)

## Kiwi Stats


# Static

def static_accepted(x = None):
    data = pd.read_excel('Trial.xlsx')
    if x is not None:
        data = data.loc[data['Tender Round'].isin(x)]
    data = data[['Tender Date', 'End Date','Tender Round', 'Reasoning', 'Tender Ref','Company Name','Tendered Unit','Status','Weekdays (From)', 'Weekdays (To)', 'Start EFA', 'End EFA','Static/Dynamic','Availability Fee (£/h)','Static Secondary']]
    data = data[data["Static/Dynamic"] == 'Static']
    #result = data[data['Status'] == 'Accepted']
    result = data[data['Reasoning'] != 'Multiple tenders received' ]

    #result_filtered = result[(result['Dynamic Primary (0.5Hz)'] != result['Dynamic High (0.5Hz)']) & (result['Dynamic Secondary (0.5Hz)'] != result['Dynamic High (0.5Hz)'])]
    result['price'] = result['Availability Fee (£/h)']/result['Static Secondary']
    result = result[result['price'] != 0]
    result['Time_dif'] = result['End Date'] - result['Tender Date']
    result['Time_dif'] = pd.to_numeric(result['Time_dif'].dt.days, downcast = 'integer')
    result = result[result['Time_dif'] <= 62]
    return result

newpath = r'/Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/Database/FFR'
os.chdir(newpath)

# maxi = result['Tender Round'].max()
#x = [124,125,126]
result_s = static_accepted(x)

df20 = result_s
df20['Start EFA'] = (pd.to_numeric(df20['Start EFA'], errors='coerce')).fillna(0).astype(int)
df20['End EFA'] = (pd.to_numeric(df20['End EFA'], errors='coerce')).fillna(0).astype(int)
df20['blocks_nr']='test'
df20['blocks']='test'

for i in range(0, len(df20)):
    if  (df20['End EFA'].iloc[i]-df20['Start EFA'].iloc[i]==1): 
        df20['blocks'].iloc[i] =(df20['Start EFA'].iloc[i])
        df20['blocks_nr'].iloc[i]=1
    
    elif (df20['Start EFA'].iloc[i] > df20['End EFA'].iloc[i]):
        df20['blocks'].iloc[i]= range(df20['Start EFA'].iloc[i], 7)
        df20['blocks_nr'].iloc[i]=len(df20['blocks'].iloc[i])
    else:
        df20['blocks'].iloc[i]= range(df20['Start EFA'].iloc[i], df20['End EFA'].iloc[i])
        df20['blocks_nr'].iloc[i]=len(df20['blocks'].iloc[i])

df20 = df20.reindex(df20.index.repeat(df20.blocks_nr)).reset_index()
df20['EFA_block'] = 'adi'

j=0
for i in range(0,len(df20)):
    print ('==================', i)
    if (df20['blocks_nr'].loc[i]==1):
        #print('mpikame sto prwto if')
        print('First IF condition\n')
        j=0
        #print(df20['Start EFA'].loc[i], df20['End EFA'].loc[i])
        df20['EFA_block'].loc[i] = 'EFA {}'.format(df20['Start EFA'].loc[i])
        #print('>>>>>>>>>>>>>>> ',df20['EFA_block'].loc[i])
        print('>>>>>>>>>>>>>>> our EFA block is: ',df20['EFA_block'].loc[i])
    else:
#         print('mpikame sto', df20['blocks'].loc[i], ' to opoio exei length ', len(df20['blocks'].loc[i]))
#         print('poso einai to j? ', j)
        print('ELSE we are in ', df20['blocks'].loc[i], ' and its length is ', len(df20['blocks'].loc[i]))
        print('What is j: ', j)
        if(j<len(df20['blocks'].loc[i])):
#             print('to j einai: ',j)
#             print('j is: ', j)
#             print(len(df20['blocks'].loc[i]),len(df20['blocks'].loc[i]))
            print('element of the range: ', df20['blocks'].loc[i][j])
           
            df20['EFA_block'].loc[i] = 'EFA {}'.format(df20['blocks'].loc[i][j])
#             print('----------------------',df20['EFA_block'].loc[i])

            j+=1
            
        elif(j==len(df20['blocks'].loc[i])):
            j=0
            print('!!!!!!!!!: ', df20['blocks'].loc[i][j])
            df20['EFA_block'].loc[i] = 'EFA {}'.format(df20['blocks'].loc[i][j])
#             print('----------------------',df20['EFA_block'].loc[i])
            j+=1
        
        elif(j>len(df20['blocks'].loc[i])):
            j=0
            print('????????: ', df20['blocks'].loc[i][j])
            df20['EFA_block'].loc[i] = 'EFA {}'.format(df20['blocks'].loc[i][j])
#             print('----------------------',df20['EFA_block'].loc[i])
            j+=1
        else:
            j=0


df20['price'] = df20['price'].astype(float)
df20 = df20.sort_values('Status')

df20 = df20.sort_values('EFA_block')

## Static Price Ranges

Tender = 'Tender Round ' + str(df20['Tender Round'].max())
Title = Tender + ' Static Accepted & Rejected Price Ranges'
fig = px.box(df20, x="EFA_block", y = "price", color = 'Status')
fig.update_layout(title = Title, xaxis_title = Tender)
fig.layout.template = 'seaborn'
fig.show()
#plot(fig)

Month = df20["End Date"].max().strftime('%B')
year = (datetime.now() + timedelta(days=0) ).strftime('%Y')
#newpath = os.chidir("_Sent_availability_files_to_client\ " +year)
newpath = r'//Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/Monthly Meeting/' +year + '/' +Month+ '/SFFR/'

if not os.path.exists(newpath):
    os.makedirs(newpath) 
os.chdir(newpath)
#change working directory to save output file

#dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file

current_directory = os.getcwd()

#Edit the excel file
#mypath = current_directory + "/fafr.xlsx"

df20['Date'] = df20['End Date'].dt.strftime('%b-%y')
df_A = df20
df_A = df_A.sort_values(by='Date')
months = ["Jan-20", "Feb-20", "Mar-20","Apr-20", "May-20","Jun-20","Jul-20","Aug-20","Sep-20","Oct-20","Nov-20","Dec-20"]
# months1 = ["Dec-19", "Jan-20", "Feb-20", "Mar-20",
#            "Apr-20", "May-20", "Jun-20"]
df_A['Date'] = pd.Categorical(df_A['Date'], 
                                  categories=months, ordered=True)
df_A = df_A.sort_values(["Date", "EFA_block"])

#df20 = df20.sort_values(['Tender Round','EFA_block', 'Status'], ascending = ['True','True', 'True'])

#df20 = df20.sort_values(['EFA_block', 'Status'], ascending = ['True', 'True'])
fig = px.box(df_A, x = 'EFA_block',y="price", facet_col="Date", color="Status",
             boxmode="overlay",hover_name = 'Status', points = "all")
title  = 'Static FFR Monthly Results'
fig.update_layout(title = title)
fig.layout.template = 'seaborn'
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")

fig.show()

#plot(fig)

df20_grouped = df20.groupby(['EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Static Secondary', 'sum' )).reset_index()

Tender = 'Tender Round ' + str(df19['Tender Round'].max())
Title = Tender + ' Static Market Results & Volumes'
fig = px.scatter(df20_grouped, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Status")
fig.update_layout(title = Title, xaxis_title = Tender)
fig.show()
#plot(fig)

df20_grouped = df20.groupby(['Tender Round','EFA_block', 'Status', 'Date']).agg(avg_price=('price', 'median'), total_volume=('Static Secondary', 'sum' )).reset_index()

df20_grouped = df20_grouped.sort_values(by='Date')
months = ["Jan-20", "Feb-20", "Mar-20","Apr-20", "May-20","Jun-20","Jul-20","Aug-20","Sep-20","Oct-20","Nov-20","Dec-20"]
# months1 = ["Dec-19", "Jan-20", "Feb-20", "Mar-20",
#            "Apr-20", "May-20", "Jun-20"]
df20_grouped['Date'] = pd.Categorical(df20_grouped['Date'], 
                                  categories=months, ordered=True)
df20_grouped = df20_grouped.sort_values(["Date", "EFA_block"])

fig = px.scatter(df20_grouped, x = 'EFA_block',y="avg_price",size = "total_volume",hover_name = 'Status', facet_col="Date", color="Status")
title = 'Static FFR Monthly Volumes'
fig.update_layout(title = title, yaxis_title = 'Price (£/MW)')
#fig.add_scatter(x = dfk_grouped['EFA_block'], y = dfk_grouped['avg_price'])
fig.layout.template = 'seaborn'
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")

fig.show()
#plot(fig)

## Kiwi Power

dfk_s = df20[df20['Company Name'] == 'Kiwi Power Ltd']

dfk_s_grouped = dfk_s.groupby(['EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Static Secondary', 'sum' )).reset_index()

Tender = 'Tender Round ' + str(df19['Tender Round'].max())
Title = 'Kiwi Power SFFR ' + Tender + ' Results'
fig = px.scatter(dfk_s_grouped, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Status")
fig.update_layout(title = Title)
fig.show()
#plot(fig)

dfk = df19[df19['Company Name'] == 'Kiwi Power Ltd']

dfk_s_grouped = dfk_s.groupby(['Tender Round','EFA_block', 'Status', 'Date']).agg(avg_price=('price', 'median'), total_volume=('Static Secondary', 'sum' )).reset_index()

Tender = 'Tender Round ' + str(int(df19['Tender Round'].max()))
Title = 'Kiwi Power Static FFR Results'
fig = px.scatter(dfk_s_grouped, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Status", hover_name = "Status", facet_col = "Date")
fig.update_layout(title = Title)
plot(fig, filename = newpath + Title + '.html')
fig.write_image(title + ".png")
fig.show()
#plot(fig)

