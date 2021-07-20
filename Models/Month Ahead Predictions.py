#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:07:20 2020

@author: adieleejiofor
"""

import pandas as pd
import datetime
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
import statsmodels.api as sm
import calendar
import os


#pd.options.mode.chained_assignment = None

def dynamic_accepted(x = None):
    data = pd.read_excel('Trial.xlsx')
    if x is not None:
        data = data.loc[data['Tender Round'].isin(x)]
    data = data[['Tender Date','Start Date', 'End Date','Tender Round', 'Tender Ref','Company Name','Tendered Unit','Status','Weekdays (From)', 'Weekdays (To)', 'Start EFA', 'End EFA','Static/Dynamic','Availability Fee (£/h)','Dynamic Primary (0.5Hz)', 'Dynamic Secondary (0.5Hz)', 'Dynamic High (0.5Hz)']]
    data = data[data["Static/Dynamic"] == 'Dynamic']
    data = data[data['Status'] == 'Accepted']
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


#pd.options.mode.chained_assignment = None

def dynamic_accepted2(x = None):
    data = pd.read_excel('Trial2.xlsx')
    if x is not None:
        data = data.loc[data['Tender Round'].isin(x)]
    data = data[['Tender Date','Start Date', 'End Date','Tender Round', 'Tender Ref','Company Name','Tendered Unit','Status','Weekdays (From)', 'Weekdays (To)', 'Start EFA', 'End EFA','Static/Dynamic','Availability Fee (£/h)','Dynamic Primary (0.5Hz)', 'Dynamic Secondary (0.5Hz)', 'Dynamic High (0.5Hz)']]
    data = data[data["Static/Dynamic"] == 'Dynamic']
    data = data[data['Status'] == 'Accepted']
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
    result= result[result['Time_dif'] <= 62]
    return result


## Refer to the Tender round you want.

#x = [121]
result = dynamic_accepted()
result2 = dynamic_accepted2()

df19 = result
df18 = result2
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



# Reindex repeats the values from the list to enable grouping

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


## Graphing situation

df18

#df19_hg = pd.pivot_table(df19, values=['Tender Round','EFA_block'], index = None , aggfunc = np.mean ).reset_index()

df18_grouped = df18.groupby(['Tender Round']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum')).reset_index()

df19_grouped = df19.groupby(['Tender Round','EFA_block']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum')).reset_index()

df20 = df19

df20["Months"] = df20['End Date'].dt.month

# df20

## Median variable for Regression

variable_median = df18.groupby(['Tender Round']).price.median()
variable_median =  variable_median.to_frame()
variable_median.reset_index(inplace = True)
variable_median1 = variable_median.groupby(['Tender Round']).price.mean().values.reshape(-1,1)

# variable_median = variable_median.groupby(['Tender Round']).price.mean()
# variable_median = variable_median.groupby(['Company','Week TR_U', 'EFA Blocks_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()
# variable_median.reset_index(inplace = True)

variable_median1

mat = df19['Tender Round'].max() 
variable_median = variable_median[variable_median['Tender Round'] < mat]
median_variable = variable_median.groupby(['Tender Round']).price.mean().values.reshape(-1,1)

median_variable.shape

## Getting the Requirements 

df19_grouped = df19.groupby(['Tender Round','EFA_block', 'Status']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum' )).reset_index()

df19_grouped

data2 = pd.read_excel('Requirements.xlsx')
data2 = data2.drop(['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 9', 'Unnamed: 8'], axis = 1)
data2['Sum_requirement'] = data2['Primary'] + data2['Secondary'] + data2['High']
data2['Average'] = (data2['Primary'] + data2['Secondary'] + data2['High'])/3

data2.head()

mats = df19['Tender Round'].max()+1
data2 = data2[data2['Tender Round'] <= mats]

## Fit Requirements to EFA rounds

mat = df19['Tender Round'].max() 
data3 = data2[data2['Tender Round'] <= mat]
Average_Requirements = data3.groupby(['Tender Round']).Average.mean().values.reshape(-1,1)
Future_Requirements = data2.groupby(['Tender Round']).Average.mean().values.reshape(-1,1)
#Average_Requirements

## Scenario Forecasts (35th, median , 75th and 95th)


### 35th quantile

TR_Price_35 = df19.groupby(['Tender Round', 'EFA_block']).price.quantile(.35)
TR_Price_35 =  TR_Price_35.to_frame()
TR_Price_35.reset_index(inplace = True)
TR_Price_35 = TR_Price_35.groupby(['Tender Round']).price.mean()
#TR_Price_75
#TR_Price_75.to_csv('TR_Price_75.csv')

### Median

TR_Price_median = df19.groupby(['Tender Round', 'EFA_block']).price.median()
TR_Price_median =  TR_Price_median.to_frame()
TR_Price_median.reset_index(inplace = True)
TR_Price_median = TR_Price_median.groupby(['Tender Round']).price.mean()
# TR_Price_median =  TR_Price_median.to_frame()
# TR_Price_median.reset_index(inplace = True)
#TR_Price_median
#TR_Price_median.to_csv('TR_Price_median.csv')

### Max Price

TR_Price_max = df19.groupby(['Tender Round', 'EFA_block']).price.max()
TR_Price_max =  TR_Price_max.to_frame()
TR_Price_max.reset_index(inplace = True)
TR_Price_max = TR_Price_max.groupby(['Tender Round']).price.mean()
# TR_Price_max.reset_index(inplace = True)
# TR_Price_max = TR_Price_max.groupby(['Tender Round']).price.mean()

# TR_Price_max.to_csv('Maximum.csv')

### 75th quantile

TR_Price_75 = df19.groupby(['Tender Round', 'EFA_block']).price.quantile(.75)
TR_Price_75 =  TR_Price_75.to_frame()
TR_Price_75.reset_index(inplace = True)
TR_Price_75 = TR_Price_75.groupby(['Tender Round']).price.mean()
#TR_Price_75
#TR_Price_75.to_csv('TR_Price_75.csv')

TR_Price_75

#TR_Price_75_R = TR_Price_75.drop(['Tender Round'], axis = 1)

### 95th quantile

TR_Price_95 = df19.groupby(['Tender Round', 'EFA_block']).price.quantile(.95)
TR_Price_95 =  TR_Price_95.to_frame()
TR_Price_95.reset_index(inplace = True)
TR_Price_95 = TR_Price_95.groupby(['Tender Round']).price.mean()
# TR_Price_95 =  TR_Price_95.to_frame()
# TR_Price_95.reset_index(inplace = True)
#TR_Price_95
#TR_Price_95.to_csv('TR_Price_95.csv')

#TR_Price_95_R = TR_Price_95.drop(['Tender Round'], axis = 1)

df20_grouped = df20.groupby(['Tender Round','End Date']).price.median()
df20_grouped =  df20_grouped.to_frame()
df20_grouped.reset_index(inplace = True)
df20_grouped["Months"] = df20_grouped['End Date'].dt.month
# TR_Price_median = TR_Price_median.groupby(['Tender Round']).price.mean()

#df20_grouped

#df20_grouped1 = df20_grouped.groupby(['Tender Round', 'Months']).price.mean
# df20_grouped =  df20_grouped.to_frame()

## Multiple Variable column

data2["Months"] = data2['Month'].dt.month

data2_grouped = data2.groupby(['Tender Round','Months']).Average.mean()
data2_grouped =  data2_grouped.to_frame()
data2_grouped.reset_index(inplace = True)
# data2_grouped = data2_grouped[data2_grouped['Tender Round'] <= mats]

data2_grouped

dummy2 = sm.categorical(data2_grouped["Months"].values.reshape(1,-1), drop =True)
dummy2

dummy = sm.categorical(df20_grouped["Months"].values.reshape(1,-1), drop =True)
dummy

Average_Requirements.shape

median_variable.shape

dummy.shape

xData = np.hstack((dummy, Average_Requirements, median_variable))

median_variable.shape

dummy2.shape

tData = np.hstack((dummy2, Future_Requirements, variable_median1))

median_variable

model = sm.OLS(TR_Price_median, xData)
results = model.fit()
print(results.summary())


#print(results.summary())

# model = sm.OLS(TR_Price_median, tData)
# results = model.fit()
# print(results.summary())

safe = results.predict(tData)
print(safe)

### Very Safe Scenario

model2 = sm.OLS(TR_Price_35, xData)
results2 = model2.fit()

very_safe = results2.predict(tData)
print(very_safe)

### 75th percentile

model3 = sm.OLS(TR_Price_75, xData)
results3 = model3.fit()

Optimistic = results3.predict(tData)
print(Optimistic)

## Add results to Dataframe to create scenarios

median = pd.DataFrame(TR_Price_median, columns=['price'])
very_safe = pd.DataFrame(very_safe, columns=['very_safe'])
safe = pd.DataFrame(safe, columns=['Safe'])
Optimistic = pd.DataFrame(Optimistic, columns=['Optimistic'])
Maximum = pd.DataFrame(TR_Price_max, columns=['price'])
Maximum['Max'] = Maximum['price']
del Maximum['price']
Maximum = Maximum.reset_index(drop = True)
# medium = pd.DataFrame(forecast_75, columns=['Medium'])
# high = pd.DataFrame(forecast_95, columns=['High'])


median = median.reset_index(drop = True)
#median.drop(columns = 'Tender Round')
#median.reset_index(inplace = True)
#median

column = data2[['Tender Round', 'Month']]
column = column.drop_duplicates(subset=['Tender Round'])
column = column.reset_index(drop=True)
#column

frays = [column, median, very_safe, safe, Optimistic, Maximum]
#          , safe, medium, high

scenarios = pd.concat(frays, axis = 1)
# scenarios

fig = px.line(scenarios, x = 'Month',  y='price')

# Only thing I figured is - I could do this 
fig.add_scatter(x =scenarios['Month'] , y=scenarios['Safe']) # Not what is desired - need a line 
# fig.add_scatter(x =scenarios['Month'] , y=scenarios['High'])
#fig.add_scatter(x =scenarios['Month'] , y=scenarios['price'])

# Show plot 
fig.show()

# Month = df19["End Date"].max().strftime('%B') + 1
import datetime
from dateutil import relativedelta

Month = datetime.date.today() + relativedelta.relativedelta(months=2)
Month = Month.strftime('%B')
Month

# import datetime
# from datetime import date, timedelta
import datetime
import _datetime

# today = _datetime.date.today()

# from dateutil.relativedelta import *
year = (_datetime.date.today() + timedelta(days=0) ).strftime('%Y')

#newpath = os.chidir("_Sent_availability_files_to_client\ " +year)
newpath = r'/Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/FFR/Tender Scenarios/' +year + '/' +Month+ '/'

if not os.path.exists(newpath):
    os.makedirs(newpath) 
os.chdir(newpath)
#change working directory to save output file

#dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file

current_directory = os.getcwd()

#Edit the excel file
#mypath = current_directory + "/fafr.xlsx"

Title = 'DFFR Dynamic Model Predictions vs Actual'
fig = go.Figure()
fig.add_trace(go.Scatter(x=scenarios["Month"],y =scenarios["price"],
                    mode='lines+markers',
                    name='Average DFFR Accepted Price'))
fig.add_trace(go.Scatter(x=scenarios["Month"], y =scenarios["Max"],mode = 'lines+markers' ,name = "Maximum DFFR Accepted Price"))
fig.add_trace(go.Scatter(x=scenarios["Month"], y =scenarios["very_safe"],mode = 'lines+markers' ,name = "Safe Forecast"))
fig.add_trace(go.Scatter(x=scenarios["Month"], y =scenarios["Safe"],mode = 'lines+markers' ,name = "Medium Forecast"))
fig.add_trace(go.Scatter(x=scenarios["Month"], y =scenarios["Optimistic"],mode = 'lines+markers' ,name = "Aggressive Forecast"))
fig.update_layout(title = Title, xaxis_title = 'Month', yaxis_title = 'Price (£)')
fig.layout.template = 'plotly_white'
plot(fig, filename = newpath + Title + '.html')
fig.show()
# plot(fig)

scenarios.to_excel('scenarios.xlsx')

median_price = df19.groupby(['Tender Round', 'EFA_block']).agg(EFA_price=('price', 'median')).reset_index()

# median_pricer = pd.DataFrame(median_price, columns=['price'])

# median = pd.DataFrame(TR_Price_median, columns=['price'])
# median_price

Average_median =  TR_Price_median.to_frame()
Average_median.reset_index(inplace = True)


Average_median.head()

median_price.head()

scenarios_1 = scenarios[scenarios['Tender Round'] == scenarios['Tender Round'].max()]

scenarios_1

Recommendations = median_price.merge(Average_median, on=["Tender Round"], how = 'inner'  )


Recommendations['value'] = Recommendations['EFA_price']/Recommendations['price'] 

Recommendations['Kiwi EFA'] = Recommendations['price'] * Recommendations['value'] 

Recommendations1 = Recommendations[Recommendations['Tender Round'] == Recommendations['Tender Round'].max()]

Recommendations2 = Recommendations1['value']
Recommendations2 =  Recommendations2.to_frame()
Recommendations2.reset_index(inplace = True)

Recommendations3 = Recommendations1['EFA_block']
Recommendations3 =  Recommendations3.to_frame()
Recommendations3.reset_index(inplace = True)

b = float(scenarios_1['Safe'])
c = float(scenarios_1['Optimistic'])
d = float(scenarios_1['very_safe'])

Recommendations2['Safe Price per EFA'] = Recommendations2['value'] * b
Recommendations2['Very Cautious Price per EFA'] = Recommendations2['value'] * d
Recommendations2['Aggressive Price per EFA'] = Recommendations2['value'] * c

Recommendations2 = Recommendations2.drop(['index', 'value'], axis=1)
Recommendations3 = Recommendations3.drop(['index'], axis=1)

Recommendations_d = Recommendations3.merge(Recommendations2, left_index = True , right_index = True  )


# import datetime

import datetime
from dateutil.relativedelta import relativedelta
now = datetime.datetime.now()
use_date = now+relativedelta(months=+1)
a = int(calendar.monthrange(use_date.year, use_date.month)[1])

Recommendations_d['Very Cautious Revenue'] = Recommendations_d['Very Cautious Price per EFA'] * 4 * a
Recommendations_d['Safe Revenue'] = Recommendations_d['Safe Price per EFA'] * 4 * a
Recommendations_d['Aggressive Revenue'] = Recommendations_d['Aggressive Price per EFA'] * 4 * a
Recommendations_d.loc['mean'] = Recommendations_d.mean()  # adding a row
Recommendations_d.at['mean', 'EFA_block'] = 'Average Price'


Recommendations_d.to_excel('DFFR Recommendations.xlsx', index=False)

Title = 'DFFR Revenue Predictions'
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=Recommendations_d["EFA_block"],y =Recommendations_d["Safe Price per EFA"],
                    mode='lines+markers',
                    name='Safe Price Forecast'))
fig.add_trace(go.Scatter(x=Recommendations_d["EFA_block"], y =Recommendations_d["Very Cautious Price per EFA"],mode = 'lines+markers' ,name = "Very Safe Forecast"))
fig.add_trace(go.Scatter(x=Recommendations_d["EFA_block"], y =Recommendations_d["Aggressive Price per EFA"],mode = 'lines+markers' ,name = "Aggresive Forecast"))
fig.add_bar(x=Recommendations_d["EFA_block"], y =Recommendations_d["Very Cautious Revenue"], name = "Very Cautious Revenue",secondary_y = True, marker =dict(opacity = 0.6))
fig.add_bar(x=Recommendations_d["EFA_block"], y =Recommendations_d["Safe Revenue"], name = "Safe Revenue",secondary_y = True, marker =dict(opacity = 0.6))
fig.add_bar(x=Recommendations_d["EFA_block"], y =Recommendations_d["Aggressive Revenue"], name = "Aggressive Revenue",secondary_y = True, marker =dict(opacity = 0.6))
fig.update_layout(title = '<b>DFFR Revenue Predictions</b>')
fig.update_xaxes(title_text="<b>EFA Block</b>")
fig.update_yaxes(title_text="<b>Revenue (£)</b>", secondary_y=True)
fig.update_yaxes(title_text="<b>Price (£)</b>", secondary_y=False)
fig.layout.template = 'plotly_white'
plot(fig, filename = newpath + Title + '.html')
fig.show()

## Take out requirements and use requirements divided by last TR's rejected MW's

### Then these values should be used as stand ins (i.e. if things stay the same)

### Other scenarios can be formed by increasing rejected volumes over time by maybe 10% every month etc





Recommendations_d["Very Cautious Revenue"].sum()

Recommendations_d["Safe Revenue"].sum()

Recommendations_d["Aggressive Revenue"].sum()

## Static FFR Forecast

# import datetime
# from datetime import date, timedelta
# import datetime
# import _datetime

# today = _datetime.date.today()

# from dateutil.relativedelta import *
# year = (_datetime.date.today() + timedelta(days=0) ).strftime('%Y')

#newpath = os.chidir("_Sent_availability_files_to_client\ " +year)
newpath = r'/Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/Database/FFR'

if not os.path.exists(newpath):
    os.makedirs(newpath) 
os.chdir(newpath)
#change working directory to save output file

#dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file

current_directory = os.getcwd()

#Edit the excel file
#mypath = current_directory + "/fafr.xlsx"

def static_accepted(x = None):
    data = pd.read_excel('Trial.xlsx')
    if x is not None:
        data = data.loc[data['Tender Round'].isin(x)]
    data = data[['Tender Date', 'End Date','Tender Round', 'Tender Ref','Company Name','Tendered Unit','Status','Weekdays (From)', 'Weekdays (To)', 'Start EFA', 'End EFA','Static/Dynamic','Availability Fee (£/h)','Static Secondary']]
    data = data[data["Static/Dynamic"] == 'Static']
    result = data[data['Status'] == 'Accepted']

    #result_filtered = result[(result['Dynamic Primary (0.5Hz)'] != result['Dynamic High (0.5Hz)']) & (result['Dynamic Secondary (0.5Hz)'] != result['Dynamic High (0.5Hz)'])]
    result['price'] = result['Availability Fee (£/h)']/result['Static Secondary']
    #result = result[result['price'] != 0]
    result['Time_dif'] = result['End Date'] - result['Tender Date']
    result['Time_dif'] = pd.to_numeric(result['Time_dif'].dt.days, downcast = 'integer')
    result = result[result['Time_dif'] <= 62]
    return result

#pd.options.mode.chained_assignment = None
def static_accepted2(x = None):
    data = pd.read_excel('Trial2.xlsx')
    if x is not None:
        data = data.loc[data['Tender Round'].isin(x)]
    data = data[['Tender Date', 'End Date','Tender Round', 'Tender Ref','Company Name','Tendered Unit','Status','Weekdays (From)', 'Weekdays (To)', 'Start EFA', 'End EFA','Static/Dynamic','Availability Fee (£/h)','Static Secondary']]
    data = data[data["Static/Dynamic"] == 'Static']
    result = data[data['Status'] == 'Accepted']
    
        #result_filtered = result[(result['Dynamic Primary (0.5Hz)'] != result['Dynamic High (0.5Hz)']) & (result['Dynamic Secondary (0.5Hz)'] != result['Dynamic High (0.5Hz)'])]
    result['price'] = result['Availability Fee (£/h)']/result['Static Secondary']
    #result = result[result['price'] != 0]
    result['Time_dif'] = result['End Date'] - result['Tender Date']
    result['Time_dif'] = pd.to_numeric(result['Time_dif'].dt.days, downcast = 'integer')
    result = result[result['Time_dif'] <= 62]
    return result


result_S = static_accepted()
result_S2 = static_accepted2()

df21 = result_S
df22 = result_S2
df21['Start EFA'] = (pd.to_numeric(df21['Start EFA'], errors='coerce')).fillna(0).astype(int)
df21['End EFA'] = (pd.to_numeric(df21['End EFA'], errors='coerce')).fillna(0).astype(int)
df21['blocks_nr']='test'
df21['blocks']='test'

for i in range(0, len(df21)):
    if  (df21['End EFA'].iloc[i]-df21['Start EFA'].iloc[i]==1): 
        df21['blocks'].iloc[i] =(df21['Start EFA'].iloc[i])
        df21['blocks_nr'].iloc[i]=1
    
    elif (df21['Start EFA'].iloc[i] > df21['End EFA'].iloc[i]):
        df21['blocks'].iloc[i]= range(df21['Start EFA'].iloc[i], 7)
        df21['blocks_nr'].iloc[i]=len(df21['blocks'].iloc[i])
    else:
        df21['blocks'].iloc[i]= range(df21['Start EFA'].iloc[i], df21['End EFA'].iloc[i])
        df21['blocks_nr'].iloc[i]=len(df21['blocks'].iloc[i])

df21 = df21.reindex(df21.index.repeat(df21.blocks_nr)).reset_index()
df21['EFA_block'] = 'adi'

j=0
for i in range(0,len(df21)):
   # print ('==================', i)
    if (df21['blocks_nr'].loc[i]==1):
        #print('mpikame sto prwto if')
       # print('First IF condition\n')
        j=0
        #print(df20['Start EFA'].loc[i], df20['End EFA'].loc[i])
        df21['EFA_block'].loc[i] = 'EFA {}'.format(df21['Start EFA'].loc[i])
        #print('>>>>>>>>>>>>>>> ',df20['EFA_block'].loc[i])
        #print('>>>>>>>>>>>>>>> our EFA block is: ',df21['EFA_block'].loc[i])
    else:
#         print('mpikame sto', df20['blocks'].loc[i], ' to opoio exei length ', len(df20['blocks'].loc[i]))
#         print('poso einai to j? ', j)
        #print('ELSE we are in ', df21['blocks'].loc[i], ' and its length is ', len(df21['blocks'].loc[i]))
       # print('What is j: ', j)
        if(j<len(df21['blocks'].loc[i])):
#             print('to j einai: ',j)
#             print('j is: ', j)
#             print(len(df20['blocks'].loc[i]),len(df20['blocks'].loc[i]))
#            print('element of the range: ', df21['blocks'].loc[i][j])
           
            df21['EFA_block'].loc[i] = 'EFA {}'.format(df21['blocks'].loc[i][j])
#             print('----------------------',df20['EFA_block'].loc[i])

            j+=1
            
        elif(j==len(df21['blocks'].loc[i])):
            j=0
#            print('!!!!!!!!!: ', df21['blocks'].loc[i][j])
            df21['EFA_block'].loc[i] = 'EFA {}'.format(df21['blocks'].loc[i][j])
#             print('----------------------',df20['EFA_block'].loc[i])
            j+=1
        
        elif(j>len(df21['blocks'].loc[i])):
            j=0
#            print('????????: ', df21['blocks'].loc[i][j])
            df21['EFA_block'].loc[i] = 'EFA {}'.format(df21['blocks'].loc[i][j])
#             print('----------------------',df20['EFA_block'].loc[i])
            j+=1
        else:
            j=0


df21['price'] = df21['price'].astype(float)

df21["Months"] = df21['End Date'].dt.month

EFA_Price_S = df21.groupby(['Tender Round', 'EFA_block']).price.median()
Static_Median = EFA_Price_S.to_frame()
Static_Median.reset_index(inplace = True)
Static_Median

TR_Price_max_s = df21.groupby(['Tender Round', 'EFA_block']).price.max()
TR_Price_max_s =  TR_Price_max_s.to_frame()
TR_Price_max_s.reset_index(inplace = True)
TR_Price_max_s = TR_Price_max_s.groupby(['Tender Round']).price.mean()

df22['price'] = df22['price'].astype(float)
variable_median_S = df22.groupby(['Tender Round']).price.median()
variable_median_S =  variable_median_S.to_frame()
variable_median_S.reset_index(inplace = True)
variable_median_S1 = variable_median_S.groupby(['Tender Round']).price.mean().values.reshape(-1,1)

# variable_median = variable_median.groupby(['Tender Round']).price.mean()
# variable_median = variable_median.groupby(['Company','Week TR_U', 'EFA Blocks_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()
# variable_median.reset_index(inplace = True)



variable_median_S1

# mat = df19['Tender Round'].max() 
# variable_median = variable_median[variable_median['Tender Round'] < mat]
# median_variable = variable_median.groupby(['Tender Round']).price.mean().values.reshape(-1,1)

# median_variable1.shape

mat = df21['Tender Round'].max() 
variable_median_S = variable_median_S[variable_median_S['Tender Round'] < mat]
median_variable_S = variable_median_S.groupby(['Tender Round']).price.mean().values.reshape(-1,1)

median_variable_S.shape

Static_Median = Static_Median.groupby(['Tender Round']).price.mean()
Static_Median

df21_grouped = df21.groupby(['Tender Round','End Date']).price.median()
df21_grouped =  df21_grouped.to_frame()
df21_grouped.reset_index(inplace = True)
df21_grouped["Months"] = df21_grouped['End Date'].dt.month

data_s = pd.read_excel('Requirements.xlsx', sheet_name = 'Static')

mats = df21['Tender Round'].max()+1
mat = df21['Tender Round'].max()
data_s = data_s[data_s['Tender Round']<= mats]
data4 = data_s[data_s['Tender Round'] <= mat]
Average_Requirements_s = data4.groupby(['Tender Round']).Secondary.mean().values.reshape(-1,1)
Future_Requirements_s = data_s.groupby(['Tender Round']).Secondary.mean().values.reshape(-1,1)

data_s["Months"] = data_s['Month'].dt.month

data_s_grouped = data_s.groupby(['Tender Round','Months']).Secondary.mean()
data_s_grouped =  data_s_grouped.to_frame()
data_s_grouped.reset_index(inplace = True)

dummy3 = sm.categorical(data_s_grouped["Months"].values.reshape(1,-1), drop =True)
dummy3

dummy4 = sm.categorical(df21_grouped["Months"].values.reshape(1,-1), drop =True)
dummy4

Average_Requirements_s.shape

median_variable_S.shape

dummy4.shape

x2Data = np.hstack((dummy4, Average_Requirements_s, median_variable_S))

t2Data = np.hstack((dummy3, Future_Requirements_s, variable_median_S1))

model = sm.OLS(Static_Median, x2Data)
results = model.fit()
print(results.summary())

safe2 = results.predict(t2Data)
print(safe2)

median_s = pd.DataFrame(Static_Median, columns=['price'])
median_s.reset_index(inplace = True)
safe_s = pd.DataFrame(safe2, columns=['Safe'])
max_s = pd.DataFrame(TR_Price_max_s, columns=['price'])
max_s['Max'] = max_s['price']
del max_s['price']
max_s = max_s.reset_index(drop = True)

#safe_s['Tender Round'] = median_s['Tender Round']
# medium = pd.DataFrame(forecast_75, columns=['Medium'])
# high = pd.DataFrame(forecast_95, columns=['High'])


median_s = median_s.reset_index(drop = True)
#median.drop(columns = 'Tender Round')
#median.reset_index(inplace = True)
#median

# median_s

s_s = data_s.groupby(['Tender Round']).Secondary.mean()
s_s = pd.DataFrame(s_s, columns=['Secondary'])
s_s.reset_index(inplace = True)

# s_s

safe_s['Tender Round'] = s_s['Tender Round']

column_s = data_s[['Tender Round', 'Month']]
column_s = column.drop_duplicates(subset=['Tender Round'])
column_s = column.reset_index(drop=True)
#column

scenarios_s = pd.merge(left=column_s, right=safe_s, left_on = 'Tender Round', right_on = 'Tender Round')
#scenarios_s = pd.merge(left=scenarios_s, right=median_s,left_on = 'Tender Round', right_on = 'Tender Round')

frais = (scenarios_s, median_s, max_s)
scenarios_s = pd.concat(frais, axis = 1)

scenarios_s = scenarios_s.loc[:,~scenarios_s.columns.duplicated()]



scenarios_s 

# import datetime
# from datetime import date, timedelta
import datetime
import _datetime

# today = _datetime.date.today()

# from dateutil.relativedelta import *
year = (_datetime.date.today() + timedelta(days=0) ).strftime('%Y')

#newpath = os.chidir("_Sent_availability_files_to_client\ " +year)
newpath = r'/Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/FFR/Tender Scenarios/' +year + '/' +Month+ '/'

if not os.path.exists(newpath):
    os.makedirs(newpath) 
os.chdir(newpath)
#change working directory to save output file

#dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file

current_directory = os.getcwd()

#Edit the excel file
#mypath = current_directory + "/fafr.xlsx"

Title = 'FFR Static Model Predictions vs Actual'
fig = go.Figure()
fig.add_trace(go.Scatter(x=scenarios_s["Month"],y =scenarios_s["price"],
                    mode='lines+markers',
                    name='Actual Price'))
fig.add_trace(go.Scatter(x=scenarios_s["Month"], y =scenarios_s["Max"],mode = 'lines+markers' ,name = "Maximum Price"))
fig.add_trace(go.Scatter(x=scenarios_s["Month"], y =scenarios_s["Safe"],mode = 'lines+markers' ,name = "Forecast"))
fig.update_layout(title = Title, xaxis_title = 'Month', yaxis_title = 'Price (£)')
fig.layout.template = 'plotly_white'
fig.show()
plot(fig, filename = newpath + Title + '.html')
# plot(fig)

median_priceS = df21.groupby(['Tender Round', 'EFA_block']).agg(EFA_price=('price', 'median')).reset_index()

# median_pricer = pd.DataFrame(median_price, columns=['price'])

# median = pd.DataFrame(TR_Price_median, columns=['price'])
# median_price

Average_median_S =  Static_Median.to_frame()
Average_median_S.reset_index(inplace = True)


Average_median_S.head()

median_priceS.head()

scenarios_2 = scenarios_s[scenarios_s['Tender Round'] == scenarios_s['Tender Round'].max()]

Recommendations_s = median_priceS.merge(Average_median_S, on=["Tender Round"], how = 'inner'  )


Recommendations_s['value'] = Recommendations_s['EFA_price']/Recommendations_s['price'] 

Recommendations_s['Kiwi EFA'] = Recommendations_s['price'] * Recommendations_s['value'] 

Recommendations_s1 = Recommendations_s[Recommendations_s['Tender Round'] == Recommendations_s['Tender Round'].max()]

Recommendations_s2 = Recommendations_s1['value']
Recommendations_s2 =  Recommendations_s2.to_frame()
Recommendations_s2.reset_index(inplace = True)

Recommendations_s3 = Recommendations_s1['EFA_block']
Recommendations_s3 =  Recommendations_s3.to_frame()
Recommendations_s3.reset_index(inplace = True)

b = float(scenarios_2['Safe'])
# c = float(scenarios_1['Optimistic'])

Recommendations_s2['Average Forecast per EFA'] = Recommendations_s2['value'] * b
# Recommendations_s2['Optimistic Price per EFA'] = Recommendations_s2['value'] * c

Recommendations_s2 = Recommendations_s2.drop(['index', 'value'], axis=1)
Recommendations_s3 = Recommendations_s3.drop(['index'], axis=1)

scenes = Recommendations_s3.merge(Recommendations_s2, left_index = True , right_index = True  )
scenes.loc['mean'] = scenes.mean()  # adding a row
scenes.at['mean', 'EFA_block'] = 'Average Price'

scenes.to_excel('Static Recommendations.xlsx', index=False)



## Logistic Regression

