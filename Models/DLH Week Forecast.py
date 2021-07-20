#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 13:14:43 2020

@author: adieleejiofor
"""

import pandas as pd
from matplotlib import pyplot as plt
from datetime import date, timedelta
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
from datetime import timedelta
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
import statsmodels.api as sm
import calendar
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import math
import os

data = pd.read_excel('Results Analysis_A.xlsx', sheet_name = "Results by Unit")
data1 = pd.read_excel('Results Analysis_A.xlsx')

data_2 = data[data["Service_U"] == "DLH"]
data_2['Date'] = data_2['EFA Date_U'].dt.strftime('%b-%y')
data_2['Date_2'] = data_2['EFA Date_U'].astype(str).str[:-3]


data_2["Month_no"] = data_2['EFA Date_U'].dt.month
#data_2 = data_2[data_2["Week TR_U"] == "Week 36"]
#data_l

data1_d = data1[data1["Service"] == "DLH"]
#data1_d = data1_d[data1_d["Week TR"] == "Week 36"]
#data2_l

data_2.columns

data1_d_grouped = data1_d.groupby(['Week TR' ,'Month', 'EFA Blocks','EFA Date', 'Day ']
                                 ).agg(avg_price=('Clearing Price', 'mean')
                                      ).reset_index()


data1_d_grouped['Day'] = data1_d_grouped['Day ']

data_2_grouped = data_2.groupby(['Company','Week TR_U', 'EFA Blocks_U', 'Day_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()

data_2_grouped['Avg_vol'] = round(data_2_grouped['Tot_vol']/7)
data_2_grouped["EFA Blocks"] = data_2_grouped["EFA Blocks_U"]
data_2_grouped["Week TR"] = data_2_grouped["Week TR_U"]
data_2_grouped["Day"] = data_2_grouped["Day_U"]

datar2 = data_2_grouped.merge(data1_d_grouped, on=["EFA Blocks", "Week TR", "Day"], how = 'inner'  )
#datar2 = pd.merge(data_2_grouped, data1_d_grouped,  how='left', left_on=['A_c1','c2'], right_on = ['B_c1','c2'])

# Months = ["January", "February", "March"]
# datarr2 = datar2[datar2.Month.isin(Months)]


datar2.head()


# fig = px.bar(datar, x="EFA Blocks", y="Avg_vol", color='Company')
# fig.add_trace(px.scatter(datar, y = "avg_price"))
# # fig = make_subplots(specs=[[{"secondary_y": True}]])
# # fig.add_trace(go.Scatter(y=datar["avg_price"]), secondary_y=True)
# fig.show()

data_2['Week'] = data_2['EFA Date_U'].dt.week

data_2.columns

data_2 = data_2.sort_values(by='EFA Date_U')


data_3 = data_2[
    ['EFA_U', 'Clearing Price_U','Week', 'Month_no']]

data_3

data_2

data_4 = data_2.groupby(['EFA Blocks_U','Week','Week TR_U','Month_no','Day_U']).agg(Price=('Clearing Price_U', 'mean')).reset_index()

data_5 = data_4
data_6 = data_4

data_6 = data_6.sort_values(by= 'Week')

day = ["Saturday", "Sunday", "Monday", "Tuesday","Wednesday", "Thursday","Friday"]

data_6['Day_U'] = pd.Categorical(data_6['Day_U'],
                                categories=day, ordered =True)

data_6 = data_6.sort_values(
    ['Week TR_U', 'Day_U', 'EFA Blocks_U'], ascending=
                            [True, True, True])

data_5['Day_U'] = pd.Categorical(data_5['Day_U'],
                                categories=day, ordered =True)

data_5 = data_5.sort_values(
    ['Week TR_U', 'Day_U', 'EFA Blocks_U'], ascending=
                            [True, True, True])


data_6['Day_U'] = data_6['Day_U'].astype(str)
data_5['Day_U'] = data_5['Day_U'].astype(str)

# if (data_6.iloc[-1, data_6.columns.get_loc("Month_no")] ==1):
#     a= 1
# else:
#     a = (data_6.iloc[-1, data_6.columns.get_loc("Month_no")] +1)  
# b = 'EFA 1'
# c = 'EFA 2'
# d = 'EFA 3'
# e = 'EFA 4'
# f = 'EFA 5'
# g = 'EFA 6'
# h = 'Saturday'
# i = 'Sunday'
# j = 'Monday'
# k = 'Tuesday'
# l = 'Wednesday'
# m = 'Thursday'
# n = 'Friday'
# o = data_6['Week TR_U'].max() +1

# df1 =  pd.DataFrame({"EFA Blocks_U":[b,c,d,e,f,g], "Week TR_U":[o,o,o,o,o,o], "Month_no": [a,a,a,a,a,a], "Day_U" : [h,i,j,k,l,m,n]}) 

df = data_6.tail(42)

df['Week'] = df['Week'] + 1

df['Week n'] = df['Week TR_U'].astype(str).str[-2:].astype(np.int64) + 1


df['Week l'] = 'Week'

df['Week TR_U'] = df['Week l'] + " " + df['Week n'].astype(str)

del df['Week n']
del df['Week l']

df

data_6 = data_6.append(df,ignore_index=True)
data_6

# data_4 = pd.get_dummies(data_4, columns=['Month_no'])
# data_4 = pd.get_dummies(data_4, columns=['EFA Blocks_U'])
# data_4 = pd.get_dummies(data_4, columns=['Week'])
# data_4 = pd.get_dummies(data_4, columns=['Day_U'])
# data_4 = data_4.drop('Week TR_U', axis = 1)

data_7 = data_6
data_7 = pd.get_dummies(data_7, columns=['Month_no'])
data_7 = pd.get_dummies(data_7, columns=['EFA Blocks_U'])
data_7 = pd.get_dummies(data_7, columns=['Week'])
data_7 = pd.get_dummies(data_7, columns=['Day_U'])
data_7 = pd.get_dummies(data_7, columns=['Week TR_U'])

# data_7 = data_7.drop('Week TR_U', axis = 1)
# data_7 = data_7.drop('Week', axis = 1)

data_7



X = data_7.drop('Price', axis = 1)
Y = data_7['Price']
X_train, x_test, Y_train, y_test = train_test_split(X,Y, test_size=0.2)



data_7

## Linear

linear_model = LinearRegression()
# linear_model.fit(X_train, Y_train)
linear_model.fit(X, Y)

linear_model.score(X_train, Y_train)
linear_model.score(X, Y)

linear_model.coef_

predictors = X_train.columns
coef = pd.Series(linear_model.coef_,predictors).sort_values()

print(coef)

y_predict = linear_model.predict(X)

price = pd.DataFrame(y_predict, columns=['Predictions'])

y_g = price.tail(42)
y_y = Y.tail(42)

y_y



r_square = linear_model.score(x_test, y_test)
r_square

###### Mean Square Error

# linear_model_mse = mean_squared_error(y_predict, y_test)
# linear_model_mse

# math.sqrt(linear_model_mse)

Week = data_6["Week TR_U"].max()
year = (datetime.now() + timedelta(days=0) ).strftime('%Y')
month = (datetime.now() + timedelta(days=0) ).strftime('%B')
#newpath = os.chidir("_Sent_availability_files_to_client\ " +year)
newpath = r'/Users/kostas/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/FAFR/Weekly Analysis/' +year+ '/' +month+ '/' +Week+ '/'

if not os.path.exists(newpath):
    os.makedirs(newpath) 
os.chdir(newpath)
#change working directory to save output file

#dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file

current_directory = os.getcwd()


### Values on Plotly

safe = pd.DataFrame(price, columns=['Predictions'])


column = data_6[['Day_U', 'Week TR_U','EFA Blocks_U']]
# column = column.drop_duplicates(subset=['Tender Round'])
column = column.reset_index(drop=True)

column

frays = [column, safe]


scenarios = pd.concat(frays, axis = 1)


scenarios['comb'] = scenarios['Day_U'] + " " + scenarios['Week TR_U'] + " " + scenarios['EFA Blocks_U']
data_6['comb'] = data_6['Day_U'] + " " + data_6['Week TR_U'] + " " + data_6['EFA Blocks_U']

week = scenarios.tail(42)

lat = data_6.tail(42)

art = scenarios.tail(84)
# ant = art.head(42)

Title = 'Fast Acting Dynamic Model Predictions vs Actual'
fig = go.Figure()
fig.add_trace(go.Scatter(x=week["comb"],y =week["Predictions"],
                    mode='lines+markers',
                    name='Predictions'))
fig.add_trace(go.Scatter(x=lat["comb"], y =lat["Price"],mode = 'lines+markers' ,name = "Clearing Price"))
fig.add_trace(go.Scatter(x=lat["comb"], y =art["Predictions"],mode = 'lines+markers' ,name = "Previous Week"))
fig.layout.template = 'plotly_white'
fig.update_layout(title = Title, xaxis_title = 'Week (EFA Block)', yaxis_title = 'Price (£)')
fig.show()
plot(fig, filename = newpath + Title + '.html')
fig.write_image(Title + ".png")

week1 = week.drop('comb', axis=1)

week1.to_csv('Week predictions.csv')