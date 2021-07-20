#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 14:32:53 2020

@author: adieleejiofor
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
#import cufflinks as cf
import numpy as np
from datetime import datetime
from datetime import date, timedelta
import os
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


Months = [(datetime.now() - timedelta(days=15) ).strftime('%B')]
Months1 = [(datetime.now() - timedelta(days=45) ).strftime('%B')]
Months2 = [(datetime.now() - timedelta(days=75) ).strftime('%B')]

Months = Months2 + Months1 + Months
Months
Month = (datetime.now() + timedelta(days=0) ).strftime('%B')


data = pd.read_excel('Results Analysis_A.xlsx', sheet_name = "Results by Unit")
data1 = pd.read_excel('Results Analysis_A.xlsx')

## LFS Stats

data_l = data[data["Service_U"] == "LFS"]
#data_l = data_l[data_l["Week TR_U"] == data["Week TR_U"].max()]
#data_l

data1_l = data1[data1["Service"] == "LFS"]
#data1_l = data1_l[data1_l["Week TR"] == data1["Week TR"].max()]
#data1_l

data1_l_grouped = data1_l.groupby(['Week TR','Month','EFA Blocks', 'EFA Date']).agg(avg_price=('Clearing Price', 'mean')).reset_index()

data_l_grouped = data_l.groupby(['Company','Week TR_U', 'EFA Blocks_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()

data_l_grouped['Avg_vol'] = round(data_l_grouped['Tot_vol']/7)
data_l_grouped["EFA Blocks"] = data_l_grouped["EFA Blocks_U"]
data_l_grouped["Week TR"] = data_l_grouped["Week TR_U"]

datar = data_l_grouped.merge(data1_l_grouped, on=["EFA Blocks", "Week TR"], how = 'inner')


datarr = datar[datar.Month.isin(Months)]


datarr['Date'] = datarr['EFA Date'].dt.strftime('%b-%y')
datarr['Date_no'] = pd.to_datetime(datarr['EFA Date']).dt.to_period('M')




# fig = px.bar(datar, x="EFA Blocks", y="Avg_vol", color='Company')
# fig.add_trace(px.scatter(datar, y = "avg_price"))
# # fig = make_subplots(specs=[[{"secondary_y": True}]])
# # fig.add_trace(go.Scatter(y=datar["avg_price"]), secondary_y=True)
# fig.show()

datar_grouped = datarr.groupby(['Date','EFA Blocks','Company']).agg(avg_price=('avg_price', 'mean'), Avg_vol=('Avg_vol', 'mean')).reset_index()
#datar_grouped = datar_grouped.sort_values(['Week TR','EFA Blocks'], ascending = ['True','True'])

datar_grouped = datar_grouped.sort_values(by='Date')
months = ["Jan-20", "Feb-20", "Mar-20","Apr-20", "May-20","Jun-20","Jul-20","Aug-20","Sep-20","Oct-20","Nov-20","Dec-20"]
# months1 = ["Dec-19", "Jan-20", "Feb-20", "Mar-20",
#            "Apr-20", "May-20", "Jun-20"]
datar_grouped['Date'] = pd.Categorical(datar_grouped['Date'], 
                                  categories=months, ordered=True)
datar_grouped = datar_grouped.sort_values(["Date", "EFA Blocks"])

# Month = datarr["Date_no"].max().strftime('%B')

year = (datetime.now() + timedelta(days=0) ).strftime('%Y')
#newpath = os.chidir("_Sent_availability_files_to_client\ " +year)
newpath = r'//Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/Monthly Meeting/' +year + '/' +Month+ '/FAFR/'

if not os.path.exists(newpath):
    os.makedirs(newpath) 
os.chdir(newpath)
#change working directory to save output file

#dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file

current_directory = os.getcwd()

#Edit the excel file
#mypath = current_directory + "/fafr.xlsx"

fig = px.bar(datar_grouped, x = 'EFA Blocks',y="Avg_vol", facet_col="Date", color="Company",hover_name = 'Company')
title = Month + ' LFS Trends'
fig.update_layout(title = title)
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
fig.update_layout(barmode='stack')
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")
fig.show()
#plot(fig)

fig = px.scatter(datar_grouped, x = 'EFA Blocks',y="avg_price", size = "Avg_vol", facet_col="Date", color="Company",hover_name = 'Company')
title = 'Fast Acting LFS Prices and Volumes'
fig.update_layout(title = title, yaxis_title = 'Price (£/MW)')
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
fig.update_layout(barmode='stack')
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")
fig.show()
# plot(fig)

# DLH

data_2 = data[data["Service_U"] == "DLH"]
#data_2 = data_2[data_2["Week TR_U"] == "Week 36"]
#data_l

data1_d = data1[data1["Service"] == "DLH"]
#data1_d = data1_d[data1_d["Week TR"] == "Week 36"]
#data2_l

data1_d_grouped = data1_d.groupby(['Week TR' ,'Month', 'EFA Blocks','EFA Date']).agg(avg_price=('Clearing Price', 'mean')).reset_index()

data_2_grouped = data_2.groupby(['Company','Week TR_U', 'EFA Blocks_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()

data_2_grouped['Avg_vol'] = round(data_2_grouped['Tot_vol']/7)
data_2_grouped["EFA Blocks"] = data_2_grouped["EFA Blocks_U"]
data_2_grouped["Week TR"] = data_2_grouped["Week TR_U"]

datar2 = data_2_grouped.merge(data1_d_grouped, on=["EFA Blocks", "Week TR"], how = 'inner'  )
#datar2 = pd.merge(data_2_grouped, data1_d_grouped,  how='left', left_on=['A_c1','c2'], right_on = ['B_c1','c2'])

datarr2 = datar2[datar2.Month.isin(Months)]


datarr2['Date'] = datarr2['EFA Date'].dt.strftime('%b-%y')
#datarr['Date'] =pd.to_datetime(datarr.Date)


datarr2.head()

datar22_grouped = datarr2.groupby(['Date','EFA Blocks','Company', 'Week TR']).agg(avg_price=('avg_price', 'mean'), Avg_vol=('Avg_vol', 'mean')).reset_index()
#datar_grouped = datar_grouped.sort_values(['Week TR','EFA Blocks'], ascending = ['True','True'])
datarr2_grouped = datar22_grouped.groupby(['Date','EFA Blocks','Company']).agg(avg_price=('avg_price', 'mean'), Avg_vol=('Avg_vol', 'mean')).reset_index()
#datar_grouped = datar_grouped.sort_values(['Week TR','EFA Blocks'], ascending = ['True','True'])

datarr2_grouped = datarr2_grouped.sort_values(by='Date')
months = ["Jan-20", "Feb-20", "Mar-20","Apr-20", "May-20","Jun-20","Jul-20","Aug-20","Sep-20","Oct-20","Nov-20","Dec-20"]
# months1 = ["Dec-19", "Jan-20", "Feb-20", "Mar-20",
#            "Apr-20", "May-20", "Jun-20"]
datarr2_grouped['Date'] = pd.Categorical(datarr2_grouped['Date'], 
                                  categories=months, ordered=True)
datarr2_grouped = datarr2_grouped.sort_values(["Date", "EFA Blocks"])


fig = px.bar(datarr2_grouped, x = 'EFA Blocks',y="Avg_vol", facet_col="Date", color="Company",hover_name = 'Company')
title = 'Fast Acting DLH Volumes'
fig.update_layout(title = 'Fast Acting DLH Volumes')
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
fig.update_layout(barmode='stack')
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")

fig.show()
#plot(fig)

fig = px.scatter(datarr2_grouped, x = 'EFA Blocks',y="avg_price", size = "Avg_vol", facet_col="Date", color="Company",hover_name = 'Company')
title = 'Fast Acting DLH Prices and Volumes'
fig.update_layout(title = title, yaxis_title = 'Price (£/MW)')
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
fig.update_layout(barmode='stack')
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")

fig.show()
#plot(fig)

datar222_grouped = datarr2.groupby(['Date','EFA Blocks']).agg(avg_price=('avg_price', 'mean'), Avg_vol=('Avg_vol', 'mean')).reset_index()
datar222_grouped = datar222_grouped.sort_values(by='Date')
months = ["Jan-20", "Feb-20", "Mar-20","Apr-20", "May-20","Jun-20","Jul-20","Aug-20","Sep-20","Oct-20","Nov-20","Dec-20"]
# months1 = ["Dec-19", "Jan-20", "Feb-20", "Mar-20",
#            "Apr-20", "May-20", "Jun-20"]
datar222_grouped['Date'] = pd.Categorical(datar222_grouped['Date'], 
                                  categories=months, ordered=True)
datar222_grouped = datar222_grouped.sort_values(["Date", "EFA Blocks"])

fig = px.scatter(datar222_grouped, x = 'EFA Blocks',y="avg_price", size = "Avg_vol", facet_col="Date")
title = 'Fast Acting FFR Prices and Volumes'
fig.update_layout(title = title, yaxis_title = 'Price (£/MW)')
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")

#fig.update_layout(barmode='stack')
fig.show()
data_2['Date'] = data_2['EFA Date_U'].dt.strftime('%b-%y')
data_2_p = data_2.groupby(['Date','EFA_U', 'Day_U', 'EFA Blocks_U', 'Month_U']).agg(Price =('Clearing Price_U', 'mean'), Volume = ("Cleared Volume_U", 'sum')).reset_index()

data_2_p = data_2_p[data_2_p.Month_U.isin(Months)]
data_2_p = data_2_p.sort_values(by='Date')
months = ["Jan-20", "Feb-20", "Mar-20","Apr-20", "May-20","Jun-20","Jul-20","Aug-20","Sep-20","Oct-20","Nov-20","Dec-20"]
# months1 = ["Dec-19", "Jan-20", "Feb-20", "Mar-20",
#            "Apr-20", "May-20", "Jun-20"]
data_2_p['Date'] = pd.Categorical(data_2_p['Date'], 
                                  categories=months, ordered=True)
data_2_p = data_2_p.sort_values(["Date", "EFA Blocks_U"])

fig = px.scatter(data_2_p, x="EFA Blocks_U", y="Price", color = "Day_U", size = "Volume", facet_col="Month_U" )
fig.update_layout(title = 'Fast Acting DLH Volumes')
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
fig.update_layout(barmode='stack')
plot(fig, filename = newpath + title + '.html')
fig.write_image(title + ".png")

fig.show()
#plot(fig)

data_2_d = data_2.groupby(['Month_U','Day_U']).agg(Price =('Clearing Price_U', 'mean'), Volume = ("Cleared Volume_U", 'mean')).reset_index()
data_2_d

data_2_A = data_2.groupby(['Month_U']).agg(Price =('Clearing Price_U', 'mean'), Volume = ("Cleared Volume_U", 'mean')).reset_index()
data_2_A