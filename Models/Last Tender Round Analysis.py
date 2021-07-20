#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:20:40 2020

@author: adieleejiofor
"""

# Dynamic FFR

import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
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
x = [127]
def dynamic_accepted(x = None):
    data = pd.read_excel('Trial.xlsx')
    if x is not None:
        data = data.loc[data['Tender Round'].isin(x)]
    data = data[['Tender Date','Start Date', 'End Date', 'Reasoning','Tender Round', 'Tender Ref','Company Name','Tendered Unit','Status','Weekdays (From)', 'Weekdays (To)', 'Start EFA', 'End EFA','Static/Dynamic','Availability Fee (£/h)','Dynamic Primary (0.5Hz)', 'Dynamic Secondary (0.5Hz)', 'Dynamic High (0.5Hz)', 'Asset Type']]
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
    if (df19['blocks_nr'].loc[i]==1):
        j=0
        df19['EFA_block'].loc[i] = 'EFA {}'.format(df19['Start EFA'].loc[i])
    else:

        if(j<len(df19['blocks'].loc[i])):

            df19['EFA_block'].loc[i] = 'EFA {}'.format(df19['blocks'].loc[i][j])

            j+=1
            
        elif(j==len(df19['blocks'].loc[i])):
            j=0
            df19['EFA_block'].loc[i] = 'EFA {}'.format(df19['blocks'].loc[i][j])
            j+=1
        
        elif(j>len(df19['blocks'].loc[i])):
            j=0
            df19['EFA_block'].loc[i] = 'EFA {}'.format(df19['blocks'].loc[i][j])
            j+=1
        else:
            j=0


## Price Range

df19_grouped2 = df19[df19['Status']== 'Accepted']

df19_grouped3 = df19[df19['Status']== 'Rejected']

df19 = df19.sort_values(['EFA_block','Status'], ascending=[True, True])

mat = df19['Tender Round'].max()
#df19 = df19.sort_values('EFA_block')

data2 = pd.read_excel('Requirements.xlsx')
data2 = data2.drop(['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 9', 'Unnamed: 8'], axis = 1)
data2['Sum_requirement'] = data2['Primary'] + data2['Secondary'] + data2['High']
data2['Average'] = (data2['Primary'] + data2['Secondary'] + data2['High'])/3

data2 = data2[['Month', 'Tender Round', 'EFA_block', 'Average']]
data2= data2[data2['Tender Round'] == mat]

df19['Status_n'] = 'adi'

for i in range(0, len(df19)):
    if (df19['Status'].loc[i] == 'Accepted'):
        df19['Status_n'].loc[i]=1
    else:
        df19['Status_n'].loc[i]=2

df19_A = df19[df19['Status']== 'Accepted']
df19_R = df19[df19['Status'] == 'Rejected']

df19_1 = df19_A[df19_A['Tender Round'] == mat]
df19_108_grouped = df19_1.groupby(['Company Name','Tendered Unit','EFA_block', 'Status', 'Status_n']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()

Month = df19["End Date"].max().strftime('%B')
year = (datetime.now() + timedelta(days=0) ).strftime('%Y')
#newpath = os.chidir("_Sent_availability_files_to_client\ " +year)
newpath = r'/Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/FFR/Python Analysis/Historical Analysis/Tender Analysis/' +year + '/' +Month+ '/'

if not os.path.exists(newpath):
    os.makedirs(newpath) 
os.chdir(newpath)
#change working directory to save output file

#dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file

current_directory = os.getcwd()

#Edit the excel file
#mypath = current_directory + "/fafr.xlsx"

dfk = df19[df19['Company Name'] == 'Kiwi Power Ltd']

df19['Tender Round'].max()

dfk_grouped = dfk.groupby(['Tendered Unit','EFA_block', 'Status' ]).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()

dfk_grouped

Tender = 'Tender Round ' + str(df19['Tender Round'].max())
Title = Tender + ' Accepted & Rejected Dynamic Price Ranges'
fig = px.box(df19, x="EFA_block", y = "price", color = 'Status')
fig.add_scatter (x =dfk_grouped["EFA_block"], y =dfk_grouped["avg_price"],mode = 'markers' ,name = "Kiwi Bids", hovertext= dfk_grouped["Tendered Unit"], marker =dict(size = dfk_grouped['total_volume'] * 2))
fig.update_layout(title = Title, xaxis_title = 'EFA Block', yaxis_title = 'Average Price (£/MWh)')
fig.layout.template = 'plotly_white'
plot(fig, filename = newpath + Title + '.html')
fig.show()
fig.write_image(Title + '.png')

#plot(fig)
#fig.write_image("image.jpeg")

# tot_v = dfk_grouped['total_volume'].tolist()
# fig.add_trace(go.Scatter(x=dfk_grouped["EFA_block"], y =dfk_grouped["avg_price"],mode = 'markers+lines' ,name = "Kiwi Bids", hovertext= dfk_grouped["total_volume"], marker=dict(
#         color=['rgb(93, 164, 214)', 'rgb(255, 144, 14)',  'rgb(44, 160, 101)', 'rgb(255, 65, 54)'],
#         size= tot_v )))

## Market Results

df19_grouped = df19.groupby(['EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()

# df19_grouped

Tender = 'Tender Round ' + str(df19['Tender Round'].max())
Title = Tender + ' Dynamic Market Results & Volumes'
fig = px.scatter(df19_grouped, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Status")
fig.update_layout(title = Title, xaxis_title = 'EFA Block', yaxis_title = 'Average Price (£/Mwh)')
fig.layout.template = 'plotly_dark'
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + '.png')
fig.show()
#plot(fig)

## Grouped by asset type

df19_grouped_As = df19.groupby(['Asset Type','EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()
df19_grouped_As_A = df19_grouped_As[df19_grouped_As['Status'] == 'Accepted']
df19_grouped_As_R = df19_grouped_As[df19_grouped_As['Status'] == 'Rejected']

Tender = 'Tender Round ' + str(df19['Tender Round'].max())
Title = Tender + ' Accepted Asset Type'
fig = px.scatter(df19_grouped_As_A, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Asset Type")
fig.update_layout(title = Title, xaxis_title = 'EFA Block', yaxis_title = 'Average Price (£/MWh)')
fig.layout.template = 'plotly_dark'
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + '.png')
fig.show()
#plot(fig)


Tender = 'Tender Round ' + str(df19['Tender Round'].max())
Title = Tender + ' Rejected Asset Type'
fig = px.scatter(df19_grouped_As_R, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Asset Type")
fig.update_layout(title = Title, xaxis_title = 'EFA Block', yaxis_title = 'Average Price (£/MWh)')
fig.layout.template = 'plotly_dark'
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + '.png')
fig.show()
#plot(fig)

## Grouped by Company

df19.head()

df19_grouped_c = df19.groupby(['Company Name','EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()
df19_grouped_c_a = df19_grouped_c[df19_grouped_c['Status'] == 'Accepted']
df19_grouped_c_r = df19_grouped_c[df19_grouped_c['Status'] == 'Rejected']

df19_grouped_c = df19.groupby(['Company Name','Tendered Unit','EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()
df19_grouped_c_a = df19_grouped_c[df19_grouped_c['Status'] == 'Accepted']

Tender = 'Tender Round ' + str(df19['Tender Round'].max())
Title = Tender + ' Accepted Companies'
fig = px.scatter(df19_grouped_c_a, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Company Name")
fig.update_layout(title = Title, xaxis_title = 'EFA Block', yaxis_title = 'Average Price (£/MWh)')
fig.layout.template = 'plotly_dark'
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + '.png')
fig.show()
#plot(fig)

Tender = 'Tender Round ' + str(df19['Tender Round'].max())
Title = Tender + ' Rejected Companies'
fig = px.scatter(df19_grouped_c_r, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Company Name")
fig.update_layout(title = Title, xaxis_title = 'EFA Block', yaxis_title = 'Average Price (£/MWh)')
fig.layout.template = 'plotly_dark'
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + '.png')
fig.show()
#plot(fig)

## Company performance

df19_grouped_u = df19.groupby(['Tender Round','End Date','Company Name','EFA_block','Status']).agg(avg_price=('price', 'mean'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()
df19_grouped_u['Date'] = df19_grouped_u['End Date'].dt.strftime('%b-%y')


df19_grouped_u_a = df19_grouped_u[df19_grouped_u['Status'] == 'Accepted']


df19_grouped_u_ak = df19_grouped_u_a[df19_grouped_u_a['Company Name'] == "Kiwi Power Ltd"]
df19_grouped_u_ek = df19_grouped_u[df19_grouped_u['Company Name'] == "Kiwi Power Ltd"]

df19_grouped_u_ak

df19_grouped_u_a2 = df19_grouped_u_a
df19_grouped_u_a2['avg_price'] = df19_grouped_u_a2['avg_price'] * df19_grouped_u_a2['total_volume']
df19_grouped_u_a2 = df19_grouped_u_a2[['Date','Company Name', 'EFA_block', 'avg_price']]

df19_grouped_u2 = df19_grouped_u.groupby(['Tender Round','Date','Company Name','EFA_block']).agg(total_volume=('total_volume', 'sum' )).reset_index()


df19_grouped_u2 = df19_grouped_u2.merge(df19_grouped_u_a2, on=['Date',"Company Name", "EFA_block"], how = 'inner'  )


df19_grouped_u2['Price'] = df19_grouped_u2['avg_price']/df19_grouped_u2['total_volume']

df19_grouped_uk = df19_grouped_u2[df19_grouped_u2['Company Name'] == "Kiwi Power Ltd"]

df19_grouped_uk

#Months = ['Jan-20', 'Feb-20','Mar-20']
#Months1 = ['Apr-20', 'May-20','Jun-20', 'Jul-20']
#df19_grouped_u21 = df19_grouped_u2[df19_grouped_u2.Date.isin(Months)]
#df19_grouped_u22 = df19_grouped_u2[df19_grouped_u2.Date.isin(Months1)]

Title = Tender + ' DFFR Company Market Performance'
fig = px.scatter(df19_grouped_u2, x ="EFA_block", y ="Price", size = "total_volume", facet_col= 'Date',color= "Company Name")
fig.update_layout(title = Title, yaxis_title = 'Average Price (£/MWh)')
fig.layout.template = 'plotly_dark'
plot(fig, filename = Title + '.html')
fig.write_image(Title + '.png')
fig.show()
# plot(fig)
#plot(fig)

## Kiwi Bids

dfk = df19[df19['Company Name'] == 'Kiwi Power Ltd']

dfk_grouped = dfk.groupby(['EFA_block', 'Status', 'Tendered Unit']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()

dfk_grouped = dfk_grouped.sort_values(['Tendered Unit', 'EFA_block'], ascending=True)

Tender = 'Tender Round ' + str(df19['Tender Round'].max())
Title = 'Kiwi Power DFFR ' + Tender + ' Results'
fig = px.scatter(dfk_grouped, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Status")
fig.update_layout(title = Title, xaxis_title = Tender)
plot(fig, filename = newpath + Title + '.html')
fig.layout.template = 'plotly_white'
fig.write_image(Title + '.png')
fig.show()
#plot(fig)

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

#maxi = result['Tender Round'].max()
    #newpath = os.chidir("_Sent_availability_files_to_client\ " +year)
newpath = r'/Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/Database/FFR'

if not os.path.exists(newpath):
    os.makedirs(newpath) 
os.chdir(newpath)
#change working directory to save output file

#dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file

current_directory = os.getcwd()
#x = [126]
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
    if (df20['blocks_nr'].loc[i]==1):
        j=0
        df20['EFA_block'].loc[i] = 'EFA {}'.format(df20['Start EFA'].loc[i])
    else:
        if(j<len(df20['blocks'].loc[i])):
            print('element of the range: ', df20['blocks'].loc[i][j])
            df20['EFA_block'].loc[i] = 'EFA {}'.format(df20['blocks'].loc[i][j])
            j+=1
            
        elif(j==len(df20['blocks'].loc[i])):
            j=0
            print('!!!!!!!!!: ', df20['blocks'].loc[i][j])
            df20['EFA_block'].loc[i] = 'EFA {}'.format(df20['blocks'].loc[i][j])
            j+=1
        
        elif(j>len(df20['blocks'].loc[i])):
            j=0
            print('????????: ', df20['blocks'].loc[i][j])
            df20['EFA_block'].loc[i] = 'EFA {}'.format(df20['blocks'].loc[i][j])
            j+=1
        else:
            j=0


df20['price'] = df20['price'].astype(float)
df20 = df20.sort_values('Status')

# df20 = df20.sort_values('EFA_block')
# df20 = df20.sort_values('Status')
df20 = df20.sort_values(['EFA_block', 'Status'], ascending=[True, True])

Month = df20["End Date"].max().strftime('%B')
year = (datetime.now() + timedelta(days=0) ).strftime('%Y')
#newpath = os.chidir("_Sent_availability_files_to_client\ " +year)
newpath = r'/Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/FFR/Python Analysis/Historical Analysis/Tender Analysis/' +year + '/' +Month+ '/'

if not os.path.exists(newpath):
    os.makedirs(newpath) 
os.chdir(newpath)
#change working directory to save output file

#dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file

current_directory = os.getcwd()

#Edit the excel file
#mypath = current_directory + "/fafr.xlsx"

## Static Price Ranges

dfk_s = df20[df20['Company Name'] == 'Kiwi Power Ltd']

dfk_s_grouped = dfk_s.groupby(['EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Static Secondary', 'sum' )).reset_index()

Tender = 'Tender Round ' + str(df20['Tender Round'].max())
Title = Tender + ' Static Accepted & Rejected Price Ranges'
fig = px.box(df20, x="EFA_block", y = "price", color = 'Status')
fig.add_trace(go.Scatter(x=dfk_s_grouped["EFA_block"], y =dfk_s_grouped["avg_price"],mode = 'markers+lines' ,name = "Kiwi Bids", hovertext= "total_volume"))
fig.update_layout(title = Title, xaxis_title = Tender)
fig.layout.template = 'plotly_white'
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + '.png')
fig.show()
#plot(fig)

df20_grouped = df20.groupby(['EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Static Secondary', 'sum' )).reset_index()

Tender = 'Tender Round ' + str(df20['Tender Round'].max())
Title = Tender + ' Static Market Results & Volumes'
fig = px.scatter(df20_grouped, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Status")
fig.update_layout(title = Title, xaxis_title = Tender)
fig.layout.template = 'plotly_white'
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + '.png')
fig.show()
#plot(fig)

dfk_s = df20[df20['Company Name'] == 'Kiwi Power Ltd']

dfk_s_grouped = dfk_s.groupby(['EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Static Secondary', 'sum' )).reset_index()

Tender = 'Tender Round ' + str(df20['Tender Round'].max())
Title = 'Kiwi Power SFFR ' + Tender + ' Results'
fig = px.scatter(dfk_s_grouped, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Status")
fig.update_layout(title = Title, xaxis_title = Tender)
fig.layout.template = 'plotly_white'
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + '.png')
fig.show()
#plot(fig)

## Asset Size

df20_grouped

## Company Name

df20_grouped_c = df20.groupby(['Company Name','EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Static Secondary', 'sum' )).reset_index()
df20_grouped_c_a = df20_grouped_c[df20_grouped_c['Status'] == 'Accepted']
df20_grouped_c_r = df20_grouped_c[df20_grouped_c['Status'] == 'Rejected']

Tender = 'Tender Round ' + str(df20['Tender Round'].max())
Title = Tender + ' Static Accepted Companies'
fig = px.scatter(df20_grouped_c_a, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Company Name")
fig.update_layout(title = Title, xaxis_title = 'EFA Block', yaxis_title = 'Average Price (£/MWh)')
fig.layout.template = 'seaborn'
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + '.png')
fig.show()
#plot(fig)

Tender = 'Tender Round ' + str(df20['Tender Round'].max())
Title = Tender + ' Static Rejected Companies'
fig = px.scatter(df20_grouped_c_r, x ="EFA_block", y ="avg_price", size = "total_volume", color= "Company Name")
fig.update_layout(title = Title, xaxis_title = 'EFA Block', yaxis_title = 'Average Price (£/MWh)')
fig.layout.template = 'seaborn'
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + '.png')
fig.show()
plot(fig)

dfk_grouped.to_excel('Kiwi Dynamic Bids.xlsx')
dfk_s_grouped.to_excel('Kiwi Static Bids.xlsx')

# Create an email with details of last TR

fig.write_image("static.png")