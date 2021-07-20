#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:26:48 2020

@author: adieleejiofor
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
#import cufflinks as cf
import numpy as np
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import os
import sys
from datetime import datetime
from datetime import timedelta

data_b = pd.read_excel('Results Analysis_A.xlsx', sheet_name = "Block Orders")
data = pd.read_excel('Results Analysis_A.xlsx', sheet_name = "Results by Unit")
data1 = pd.read_excel('Results Analysis_A.xlsx')

# data.head()

data1['Week_no'] = data1['Week TR'].astype(str).str[-2:].astype(np.int64)
data['Week_no'] = data['Week TR_U'].astype(str).str[-2:].astype(np.int64)
# df['DATE'] = df['DATE'].astype(str).str[:-2].astype(np.int64)

## LFS Stats

data_l = data[data["Service_U"] == "LFS"]
multiple = data_l[data_l["Week_no"] >= data["Week_no"].max() -1]
data1_l = data1[data1["Service"] == "LFS"]
multiple_1 = data1_l[data1_l["Week_no"] >= data["Week_no"].max()-1]


data_l = data[data["Service_U"] == "LFS"]
data_l = data_l[data_l["Week TR_U"] == data["Week TR_U"].max()] 
data_k = data[data["Week TR_U"] == data["Week TR_U"].max()] 


# data_l

data1_l = data1[data1["Service"] == "LFS"]
data1_l = data1_l[data1_l["Week TR"] == data["Week TR_U"].max()] 

# data1_l

data1_l_grouped = data1_l.groupby(['Week TR', 'EFA Blocks']).agg(avg_price=('Clearing Price', 'mean')).reset_index()

data_l_grouped = data_l.groupby(['Company','Week TR_U', 'EFA Blocks_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()

data_l_grouped['Avg_vol'] = round(data_l_grouped['Tot_vol']/7)
data_l_grouped["EFA Blocks"] = data_l_grouped["EFA Blocks_U"]

datar = data_l_grouped.merge(data1_l_grouped, on="EFA Blocks", how = 'inner'  )

multiple_1 = multiple_1.groupby(['Week TR', 'EFA Blocks']).agg(avg_price=('Clearing Price', 'mean')).reset_index()

multiple = multiple.groupby(['Company','Week TR_U', 'EFA Blocks_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()

multiple['Avg_vol'] = round(multiple['Tot_vol']/7, 2)
multiple["EFA Blocks"] = multiple["EFA Blocks_U"]
multiple["Week TR"] = multiple["Week TR_U"]

multiplier = multiple.merge(multiple_1, on=["EFA Blocks", "Week TR"], how = 'inner')

Week = data["Week TR_U"].max()




# fig = px.bar(datar, x="EFA Blocks", y="Avg_vol", color='Company')
# fig.add_trace(px.scatter(datar, y = "avg_price"))
# # fig = make_subplots(specs=[[{"secondary_y": True}]])
# # fig.add_trace(go.Scatter(y=datar["avg_price"]), secondary_y=True)
# fig.show()

Week = data["Week TR_U"].max()
year = (datetime.now() + timedelta(days=0) ).strftime('%Y')
month = (datetime.now() + timedelta(days=0) ).strftime('%B')
#newpath = os.chidir("_Sent_availability_files_to_client\ " +year)
newpath = r'/Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/FAFR/Weekly Analysis/' +year+ '/' +month+ '/' +Week+ '/'

if not os.path.exists(newpath):
    os.makedirs(newpath) 
os.chdir(newpath)
#change working directory to save output file

#dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file

current_directory = os.getcwd()

#Edit the excel file
#mypath = current_directory + "/fafr.xlsx"

data_k = data_k[data_k['Company'] == 'KIWI POWER LTD']

data_k_grouped = data_k.groupby(['Company','Unit Name','Day_U','Service_U','EFA Blocks_U']).agg(Price =('Clearing Price_U', 'mean'), 
                                                               Volume = ('Cleared Volume_U', 'sum')).reset_index()

data_k_grouped = data_k_grouped.sort_values(by='Day_U')
Day = ["Saturday", "Sunday", "Monday", "Tuesday","Wednesday", "Thursday", "Friday"]

data_k_grouped['Day_U'] = pd.Categorical(data_k_grouped['Day_U'], 
                                  categories=Day, ordered=True)
data_k_grouped = data_k_grouped.sort_values(["Unit Name","Day_U", "EFA Blocks_U" ])
#df.sort_values(['a', 'b'], ascending=[True, False])
data_k_grouped.to_excel('Kiwi.xlsx')





fig = make_subplots(specs=[[{"secondary_y": True}]])
fig = px.bar(datar, x="EFA Blocks", y="Avg_vol", color='Company')
#fig.add_bar(secondary_y = False, x =datar["EFA Blocks"], y = datar["Avg_vol"], color = datar['Company'])
fig.add_scatter(x=datar["EFA Blocks"], y =datar["avg_price"], mode = 'markers+lines', name = 'Price(£)', secondary_y =False)
fig.update_layout(title = Week + ' LFS Prices and Volumes', yaxis_title = 'Average Volume/Price')
fig.layout.template = 'plotly_white'
fig.show()
#plotly.offline.
plot(fig, filename = newpath + Week + ' LFS Prices and Volumes.html', auto_open=False)
fig.write_image(Week + " LFS Prices and Volumes.png")

# plot(fig, filename = newpath + Week + ' LFS Prices and Volumes.png')
                    #.format(start = fromDate, end=toDate), auto_open=False)
#plot(fig)

# Weekly Comparison

### LFS

fig = px.bar(multiplier, x = 'EFA Blocks',y="Avg_vol", facet_col="Week TR", color="Company",hover_name = 'Company')
title = Week + ' LFS Trends'
fig.update_layout(title = title)
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
fig.update_layout(barmode='stack')
plot(fig, filename = newpath + title + '.html', auto_open=False)
fig.write_image(title + ' .png')
fig.show()
#plot(fig)

## DLH Stats

data_2 = data[data["Service_U"] == "DLH"]
multiple_2 = data_2[data_2["Week_no"] >= data["Week_no"].max() -1]
data1_2 = data1[data1["Service"] == "DLH"]
multiple_3 = data1_2[data1_2["Week_no"] >= data["Week_no"].max()-1]
data_3 = data[data["Service_U"] == "DLH"]
data_3 = data_3[data_3["Week_no"] >= data["Week_no"].max()-1]


data_2 = data[data["Service_U"] == "DLH"]
data_2 = data_2[data_2["Week TR_U"] == data_2["Week TR_U"].max()]
# data_3 = data_2[data_2["Week_no"] >= data_2["Week_no"].max()-1]
#data_l

data1_d = data1[data1["Service"] == "DLH"]
data1_d = data1_d[data1_d["Week TR"] == data1_d["Week TR"].max()]
#data2_l

data1_d_grouped = data1_d.groupby(['Week TR', 'EFA Blocks']).agg(avg_price=('Clearing Price', 'mean')).reset_index()

data_2_grouped = data_2.groupby(['Company','Week TR_U', 'EFA Blocks_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()

data_2_grouped['Avg_vol'] = round(data_2_grouped['Tot_vol']/7)
data_2_grouped["EFA Blocks"] = data_2_grouped["EFA Blocks_U"]

datar2 = data_2_grouped.merge(data1_d_grouped, on="EFA Blocks", how = 'inner'  )

multiple_3

multiple_3 = multiple_3.groupby(['Week TR', 'EFA Blocks']).agg(avg_price=('Clearing Price', 'mean')).reset_index()

multiple_2 = multiple_2.groupby(['Company','Week TR_U', 'EFA Blocks_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()

multiple_2['Avg_vol'] = round(multiple_2['Tot_vol']/7, 2)
multiple_2["EFA Blocks"] = multiple_2["EFA Blocks_U"]
multiple_2["Week TR"] = multiple_2["Week TR_U"]

multiplier_1 = multiple_2.merge(multiple_3, on=["EFA Blocks", "Week TR"], how = 'inner')
multiplier_1 = multiplier_1.sort_values(by=['EFA Blocks'])
datar2 = datar2.sort_values(by=['EFA Blocks'])

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig = px.bar(datar2, x="EFA Blocks", y="Avg_vol", color='Company')
#fig.add_bar(secondary_y = False, x =datar["EFA Blocks"], y = datar["Avg_vol"], color = datar['Company'])
fig.add_scatter(x=datar2["EFA Blocks"], y =datar2["avg_price"], mode = 'markers+lines', name = 'Price(£)', secondary_y =False)
fig.update_layout(title = Week + ' DLH Prices and Volumes',yaxis_title = 'Average Volume/Price')
fig.layout.template = 'plotly_white'
plot(fig, filename = newpath + Week + ' DLH Prices and Volumes.html', auto_open=False)
fig.write_image(Week + ' DLH Prices and Volumes.png')
fig.show()
#plot(fig)

### Weekly Comparison DLH

fig = px.bar(multiplier_1, x = 'EFA Blocks',y="Avg_vol", facet_col="Week TR", color="Company",hover_name = 'Company')
title = Week + ' DLH Trends'
fig.update_layout(title = title)
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
fig.update_layout(barmode='stack')
plot(fig, filename = newpath + title + '.html', auto_open=False)
fig.write_image(title + ' .png')
fig.show()
#plot(fig)

#fig = make_subplots(specs=[[{"secondary_y": True}]])
fig = px.bar(data_2, x="EFA_U", y="Cleared Volume_U", color='Company')
#fig.add_bar(secondary_y = False, x =datar["EFA Blocks"], y = datar["Avg_vol"], color = datar['Company'])
#fig.add_scatter(x=data_2["EFA_U"], y =data_2["Clearing Price_U"], mode = 'markers+lines', name = 'Price(£)')
fig.update_layout(title = Week + " Volumes Procured",yaxis_title = 'Average Volume', xaxis_title = "EFA")
fig.layout.template = 'plotly_dark'
plot(fig, filename = newpath + Week + ' Volumes Procured.html', auto_open=False)
fig.write_image(Week + ' Volumes Procured.png')
fig.show()
#plot(fig)

data_2_p = data_2.groupby(['EFA_U', 'Day_U', 'EFA Blocks_U']).agg(Price =('Clearing Price_U', 'mean'), Volume = ("Cleared Volume_U", 'sum')).reset_index()
#fig = make_subplots(specs=[[{"secondary_y": True}]])
fig = px.scatter(data_2_p, x="EFA Blocks_U", y="Price", color = "Day_U", size = "Volume" )
#fig.add_bar(secondary_y = False, x =datar["EFA Blocks"], y = datar["Avg_vol"], color = datar['Company'])
#fig.add_scatter(x=data_2["EFA_U"], y =data_2["Clearing Price_U"], mode = 'markers+lines', name = 'Price(£)')
fig.update_layout(title = Week + ' DLH Prices and Volumes (Day)', yaxis_title = 'Average Price', xaxis_title = 'EFA Blocks')
fig.layout.template = 'plotly_white'
plot(fig, filename = newpath + Week + ' DLH Prices and Volumes (Day).html', auto_open=False)
fig.write_image(Week + ' DLH Prices and Volumes (Day).png')
fig.show()
#plot(fig)

# data_2_d.to_csv('Prices.csv')

data_2_v = data_3.groupby(['Week_no','EFA_U','Day_U', 'EFA Blocks_U']).agg(Price =('Clearing Price_U', 'mean'), Volume = ("Cleared Volume_U", 'sum')).reset_index()
fig = px.scatter(data_2_v, x = 'EFA Blocks_U',y="Price", facet_col="Week_no", color="Day_U", size = 'Volume', hover_name = 'Day_U')
title = Week + ' DLH Daily Trends'
fig.update_layout(title = title)
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
fig.update_layout(barmode='stack')
plot(fig, filename = newpath + title + '.html', auto_open=False)
fig.write_image(title + ' .png')
fig.show()
#plot(fig)

data_2_d = data1_d.groupby(['Day ']).agg(Price =('Clearing Price', 'mean'), Volume = ("Cleared Volume", 'mean')).reset_index()
data_2_d['Price'] = data_2_d['Price'].round(2)
data_2_d

data_2_d = data_2_d.sort_values(by='Day ')
Day = ["Saturday", "Sunday", "Monday", "Tuesday","Wednesday", "Thursday", "Friday"]

data_2_d['Day '] = pd.Categorical(data_2_d['Day '], 
                                  categories=Day, ordered=True)
data_2_d = data_2_d.sort_values(["Day "])
#df.sort_values(['a', 'b'], ascending=[True, False])

data_2_d.to_csv('price ave.csv')

## Company with clearing price

data_b['Week_no'] = data_b['Week'].astype(str).str[-2:].astype(np.int64)


data_p = data_b

data_p = data_p[data_p['MarketName'] == 'DLH']
data_p = data_p[data_p["Week_no"] == data_b["Week_no"].max()] 

data_p_grouped = data_p.groupby(['MemberName','Week','EFA','EFA Block ','Status']).agg(Price=('ClearingPrice', 'mean'), Volume=('Volume', 'mean' )).reset_index()
data_pm_grouped = data_p.groupby(['MemberName','Month','EFA Block ','Status']).agg(Price=('ClearingPrice', 'mean'), Volume=('Volume', 'mean' )).reset_index()
# df19_grouped_u_a = df19_grouped_u[df19_grouped_u['Status'] == 'Accepted']


data_p_grouped = data_p_grouped.groupby(['MemberName','Week','EFA Block ','Status']).agg(Price=('Price', 'mean'), Volume=('Volume', 'mean' )).reset_index()


data_p_grouped['Volume'] = data_p_grouped['Volume']  * -1

data_p_grouped_a = data_p_grouped[data_p_grouped['Status'] == 'Executed']

data_p_grouped_a['Price'] = data_p_grouped_a['Price'] * data_p_grouped_a['Volume']

data_p_grouped_a['total_volume'] = data_p_grouped_a['Volume']
data_p_grouped_a = data_p_grouped_a.drop('Volume', axis=1)



data_p_grouped2 = data_p_grouped.groupby(['MemberName','Week','EFA Block ']
                                ).agg(Volume=('Volume', 'sum' )
                                     ).reset_index()

data_p_grouped2 = data_p_grouped2.merge(data_p_grouped_a, on=["MemberName",'Week',"EFA Block "], how = 'inner'  )


data_p_groupedk = data_p_grouped[data_p_grouped['MemberName'] == 'KIWI POWER LTD']

data_p_groupedk

data_p_grouped['MemberName'].values

data_p_grouped2['Price'] = data_p_grouped2['Price']/data_p_grouped2['Volume']
data_p_grouped2.to_excel('Weekly Performance.xlsx')

Tender = 'Week ' + str(data_p['Week_no'].max())
Title = Tender + ' Company Market Performance'
fig = px.scatter(data_p_grouped2, x ="EFA Block ", y ="Price", size = "Volume", color= "MemberName")
fig.update_layout(title = Title, xaxis_title = 'Size (MW)', yaxis_title = 'Average Price (£/MWh)')
fig.layout.template = 'plotly_dark'
plot(fig, filename = newpath + Title + '.html',auto_open=False)
fig.write_image(newpath + Title + ' .jpeg')
fig.show()

data_b.head()
data_b['Week_no'] = data_b['Week'].astype(str).str[-2:].astype(np.int64)
data_bd = data_b[data_b['MarketName'] == 'LFS']
data_bd = data_bd[data_bd['Status'] == 'Rejected']
multiple_bd = data_bd[data_bd["Week_no"] >= data_b["Week_no"].max() -1]
data_bf = data_bd[data_bd["Week_no"] == data_bd["Week_no"].max()] 

data_bf = data_bf[data_bf['Status'] == 'Rejected']
data_bf_grouped = data_bf.groupby(['MemberName','Week', 'EFA Block ', 'EFA']).agg(Tot_vol=('Volume', 'mean'), Price=('Price', 'mean')).reset_index()


multiple_bd_grouped = multiple_bd.groupby(['MemberName','Week', 'EFA Block ', 'EFA']).agg(Tot_vol=('Volume', 'mean'), Price=('Price', 'mean')).reset_index()


multiple_bd_grouped['Volume'] = multiple_bd_grouped['Tot_vol'] * -1



data_bc = data_b[data_b['MarketName'] == 'DLH']
multiple_b = data_bc[data_bc["Week_no"] >= data_b["Week_no"].max() -1]



data_b = data_b[data_b["MarketName"] == "DLH"]
data_b = data_b[data_b["Week_no"] == data_b["Week_no"].max()] 

data_br = data_b[data_b['Status'] == 'Rejected']


data_br_grouped = data_br.groupby(['MemberName','Week', 'EFA Block ', 'EFA']).agg(Tot_vol=('Volume', 'mean'), Price=('Price', 'mean')).reset_index()





data_b = data_b[data_b["MarketName"] == "DLH"]
data_b = data_b[data_b["Week_no"] == data_b["Week_no"].max()] 

data_br = data_b[data_b['Status'] == 'Rejected']


data_br_grouped = data_br.groupby(['MemberName','Week', 'EFA Block ', 'EFA']).agg(Tot_vol=('Volume', 'mean'), Price=('Price', 'mean')).reset_index()

data_br_grouped2 = data_br.groupby(['MemberName','Week', 'EFA Block ', 'Price']).agg(Tot_vol=('Volume', 'mean')).reset_index()

data_br_grouped['Volume'] = data_br_grouped['Tot_vol'] * -1

data_br_grouped2['Volume'] = data_br_grouped2['Tot_vol'] * -1

data_br_grouped2
multiple_br = multiple_b[multiple_b['Status'] == 'Rejected']
multiple_br_grouped = multiple_br.groupby(['MemberName','Week', 'EFA Block ', 'EFA']).agg(Tot_vol=('Volume', 'mean'), Price=('Price', 'mean')).reset_index()


multiple_br_grouped['Volume'] = multiple_br_grouped['Tot_vol'] * -1


multiple_br_grouped2 = multiple_br.groupby(['MemberName','Week', 'EFA Block ']).agg(Tot_vol=('Volume', 'mean'), Price=('Price', 'mean')).reset_index()
multiple_br_grouped2['Volume'] = multiple_br_grouped2['Tot_vol'] * -1


data_br_grouped2.columns

# fig = make_subplots(specs=[[{"secondary_y": True}]])
fig = px.bar(data_br_grouped, x="EFA Block ", y="Volume", color='MemberName')

#fig.add_bar(secondary_y = False, x =datar["EFA Blocks"], y = datar["Avg_vol"], color = datar['Company'])
# fig.add_scatter(x=data_br_grouped["EFA Block "], y =data_br_grouped["Price"], mode = 'markers+lines', name = 'Price(£)', secondary_y =False)
fig.update_layout(title = Week + ' Rejected Companies')
fig.layout.template = 'plotly_white'
plot(fig, filename = newpath + Week + ' Rejected Companies.html',auto_open=False)
fig.write_image(newpath + Week + ' Rejected Companies.jpeg')
fig.show()
# plot(fig)




#fig.add_bar(secondary_y = False, x =datar["EFA Blocks"], y = datar["Avg_vol"], color = datar['Company'])
# fig.add_scatter(x=data_br_grouped["EFA Block "], y =data_br_grouped["Price"], mode = 'markers+lines', name = 'Price(£)', secondary_y =False)

# plot(fig)



multiple_b.columns

multiple_b_grouped = multiple_b.groupby(['MemberName',
                                     'Week', 'EFA Block ']).agg(Tot_vol=
                                                                ('Volume', 'mean'), Price=
                                                                ('Price', 'mean')).reset_index()



multiple_bc = multiple_b[multiple_b['MarketName'] == 'DLH']
multiple_bc = multiple_bc[multiple_bc['Status'] == 'Executed']

multiple_bc.head()

multiple_bc = multiple_bc[multiple_bc['Price'] == multiple_bc['ClearingPrice']]


multiple_b_grouped2 = multiple_bc.groupby([
                                     'Week', 'EFA Block ', 'EFA','MemberName','Portfolio', 'Day']).agg(Tot_vol=
                                                                ('Volume', 'mean'), Price=
                                                                ('Price', 'max')).reset_index()

multiple_b_grouped2['Volume'] = multiple_b_grouped2['Tot_vol'] * -1

# fig = make_subplots(specs=[[{"secondary_y": True}]])
# fig = px.bar(multiple_b_grouped2, x="EFA Block ", y="Volume", color='MemberName')
fig = px.bar(multiple_b_grouped2, x="EFA", y="Volume", color='MemberName', hover_name = multiple_b_grouped2['Portfolio'] + '\n Price: '+multiple_b_grouped2['Price'].astype(str))
             # fig = px.scatter(multiple_b_grouped2, x="EFA", y="Volume", color = "MemberName", size = "Price", hover_name = 'Portfolio' )
# hovertext = (lockRevenuesDaily['Settlement Date']).astype(str) + '\n-Revenues: £ ' + round(lockRevenuesDaily['Revenues']).astype(str)
#fig.add_bar(secondary_y = False, x =datar["EFA Blocks"], y = datar["Avg_vol"], color = datar['Company'])
# fig.add_scatter(x=data_br_grouped["EFA Block "], y =data_br_grouped["Price"], mode = 'markers+lines', name = 'Price(£)', secondary_y =False)
fig.update_layout(title = Week + ' Clearing Price Companies')
fig.layout.template = 'plotly_white'
plot(fig, filename = newpath + Week + ' Clearing Price Companies.html',auto_open=False)
fig.show()
fig.write_image(newpath + Week + ' Clearing Price Companies.jpeg')
# plot(fig)

#fig = make_subplots(specs=[[{"secondary_y": True}]])
fig = px.bar(data_br_grouped, x="EFA", y="Price", color = "MemberName")
#fig.add_bar(secondary_y = False, x =datar["EFA Blocks"], y = datar["Avg_vol"], color = datar['Company'])
#fig.add_scatter(x=data_2["EFA_U"], y =data_2["Clearing Price_U"], mode = 'markers+lines', name = 'Price(£)')
fig.update_layout(title = Week + " Volumes Rejected",yaxis_title = 'Average Volume', xaxis_title = "EFA")
fig.layout.template = 'plotly_dark'
plot(fig, filename = newpath + Week + ' Volumes Rejected.html', auto_open=False)
fig.write_image(Week + ' Volumes Rejected.png')
fig.show()
#plot(fig)

fig = px.bar(multiple_br_grouped, x = 'EFA Block ',y="Volume", facet_col="Week",color = "MemberName")
                 #color="MemberName",hover_name = 'MemberName')
title = Week + ' DLH Rejected Trends'
fig.update_layout(title = title)
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
fig.update_layout(barmode='stack')
plot(fig, filename = newpath + title + '.html', auto_open=False)
fig.write_image(title + ' .png')
fig.show()

fig = px.scatter(multiple_br_grouped, x = 'EFA Block ',y="Price", facet_col="Week",color = "MemberName", size = "Volume")
                 #color="MemberName",hover_name = 'MemberName')
title = Week + ' DLH Rejected Price trends'
fig.update_layout(title = title)
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
# fig.update_layout(barmode='stack')
plot(fig, filename = newpath + title + '.html', auto_open=False)
fig.write_image(title + ' .png')
fig.show()

fig = px.scatter(multiple_bd_grouped, x = 'EFA Block ',y="Price", facet_col="Week",color = "MemberName", size = "Volume")
                 #color="MemberName",hover_name = 'MemberName')
title = Week + ' LFS Rejected Price trends'
fig.update_layout(title = title)
#fig.add_scatter(x=datarr["EFA Blocks"], y =datarr["avg_price"], mode = 'markers+lines', name = 'Price(£)')
fig.layout.template = 'seaborn'
# fig.update_layout(barmode='stack')
plot(fig, filename = newpath + title + '.html', auto_open=False)
fig.write_image(title + ' .png')
fig.show()


#fig = px.scatter(data_br_grouped, x="EFA", y="Price", color = "MemberName", size = "Volume" )
#fig.add_scatter(x=multiple_bc["EFA"], y =multiple_bc["ClearingPrice"], mode = 'markers', name = 'Clearing Price(£)', secondary_y =False)
#fig.update_layout(title = Week + ' Rejected Companies Volumes')
#fig.layout.template = 'plotly_white'
#plot(fig, filename = newpath + Week + ' Rejected Companies volumes.html')
#fig.write_image(newpath + Week + ' Rejected Companies volumes.png')
#fig.show()