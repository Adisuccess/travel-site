#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:04:25 2020

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
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
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


#x = [121]
result = dynamic_accepted()
result2 = dynamic_accepted2()

df19 = result
df18 = result2
df19['Start EFA'] = (pd.to_numeric(df19['Start EFA'], errors='coerce')).fillna(0).astype(int)
df19['End EFA'] = (pd.to_numeric(df19['End EFA'], errors='coerce')).fillna(0).astype(int)
df19['blocks_nr']='test'
df19['blocks']='test'

df18['Start EFA'] = (pd.to_numeric(df18['Start EFA'], errors='coerce')).fillna(0).astype(int)
df18['End EFA'] = (pd.to_numeric(df18['End EFA'], errors='coerce')).fillna(0).astype(int)
df18['blocks_nr']='test'
df18['blocks']='test'

# df19 = df19[df19['Tender Round']>= 112]
# df18 = df18[df18['Tender Round']>= 111]

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


for i in range(0, len(df18)):
    if  (df18['End EFA'].iloc[i]-df18['Start EFA'].iloc[i]==1): 
        df18['blocks'].iloc[i] =(df18['Start EFA'].iloc[i])
        df18['blocks_nr'].iloc[i]=1
        
    elif  (df18['End EFA'].iloc[i]-df18['Start EFA'].iloc[i]==0): 
        df18['blocks'].iloc[i] =(range(1,7))
        df18['blocks_nr'].iloc[i]= len(df18['blocks'].iloc[i])
        
    elif (df18['Start EFA'].iloc[i] > df18['End EFA'].iloc[i]):
        df18['blocks'].iloc[i]= range(df18['Start EFA'].iloc[i], 7)
        df18['blocks_nr'].iloc[i]=len(df18['blocks'].iloc[i])
    else:
        df18['blocks'].iloc[i]= range(df18['Start EFA'].iloc[i], df18['End EFA'].iloc[i])
        df18['blocks_nr'].iloc[i]=len(df18['blocks'].iloc[i])



# Reindex repeats the values from the list to enable grouping

df18 = df18.reindex(df18.index.repeat(df18.blocks_nr)).reset_index()
df18['EFA_block'] = 'adi'

j=0
for i in range(0,len(df18)):
    #print ('==================', i)
    if (df18['blocks_nr'].loc[i]==1):
        #print('mpikame sto prwto if')
        #print('First IF condition\n')
        j=0
        #print(df18['Start EFA'].loc[i], df18['End EFA'].loc[i])
        df18['EFA_block'].loc[i] = 'EFA {}'.format(df18['Start EFA'].loc[i])
        #print('>>>>>>>>>>>>>>> ',df18['EFA_block'].loc[i])
        #print('>>>>>>>>>>>>>>> our EFA block is: ',df18['EFA_block'].loc[i])
    else:
#         print('mpikame sto', df18['blocks'].loc[i], ' to opoio exei length ', len(df18['blocks'].loc[i]))
#         print('poso einai to j? ', j)
        #print('ELSE we are in ', df18['blocks'].loc[i], ' and its length is ', len(df18['blocks'].loc[i]))
        #print('What is j: ', j)
        if(j<len(df18['blocks'].loc[i])):
#             print('to j einai: ',j)
#             print('j is: ', j)
#             print(len(df18['blocks'].loc[i]),len(df18['blocks'].loc[i]))
           # print('element of the range: ', df18['blocks'].loc[i][j])
           
            df18['EFA_block'].loc[i] = 'EFA {}'.format(df18['blocks'].loc[i][j])
#             print('----------------------',df18['EFA_block'].loc[i])

            j+=1
            
        elif(j==len(df18['blocks'].loc[i])):
            j=0
            #print('!!!!!!!!!: ', df18['blocks'].loc[i][j])
            df18['EFA_block'].loc[i] = 'EFA {}'.format(df18['blocks'].loc[i][j])
#             print('----------------------',df18['EFA_block'].loc[i])
            j+=1
        
        elif(j>len(df18['blocks'].loc[i])):
            j=0
           # print('????????: ', df18['blocks'].loc[i][j])
            df18['EFA_block'].loc[i] = 'EFA {}'.format(df18['blocks'].loc[i][j])
#             print('----------------------',df18['EFA_block'].loc[i])
            j+=1
        else:
            j=0


## Graphing situation

df18

#df19_hg = pd.pivot_table(df19, values=['Tender Round','EFA_block'], index = None , aggfunc = np.mean ).reset_index()

df18_grouped = df18.groupby(['Tender Round', 'EFA_block']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum')).reset_index()

df19_grouped = df19.groupby(['Tender Round','EFA_block']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum')).reset_index()

df20 = df19

df20["Months"] = df20['End Date'].dt.month

# df20

## Median variable for Regression

variable_median = df18.groupby(['Tender Round', 'EFA_block']).price.median()
variable_median =  variable_median.to_frame()
variable_median.reset_index(inplace = True)
variable_median1  = pd.get_dummies(variable_median, columns=['EFA_block'])

variable_median1 = variable_median.groupby(['Tender Round', 'EFA_block']).price.mean().values.reshape(-1,1)

# variable_median = variable_median.groupby(['Tender Round']).price.mean()
# variable_median = variable_median.groupby(['Company','Week TR_U', 'EFA Blocks_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()
# variable_median.reset_index(inplace = True)

variable_median1

mat = df19['Tender Round'].max() 
variable_median = variable_median[variable_median['Tender Round'] < mat]
median_variable = variable_median.groupby(['Tender Round','EFA_block']).price.mean().values.reshape(-1,1)

variable_median

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
mat = df19['Tender Round'].max() 
data2 = data2[data2['Tender Round'] <= mats]
data3 = data2[data2['Tender Round'] <= mat]


# data2 = data2[data2['Tender Round'] >= 112]
# data3 = data3[data3['Tender Round'] >= 112]


## Fit Requirements to EFA rounds

Average_Requirements = data3.groupby(['Tender Round', 'EFA_block']).Average.mean().values.reshape(-1,1)
Future_Requirements = data2.groupby(['Tender Round','EFA_block']).Average.mean().values.reshape(-1,1)
#Average_Requirements

## Scenario Forecasts (35th, median , 75th and 95th)


### 35th quantile

TR_Price_35 = df19.groupby(['Tender Round', 'EFA_block']).price.quantile(.15)
TR_Price_35 =  TR_Price_35.to_frame()
TR_Price_35.reset_index(inplace = True)
TR_Price_35 = TR_Price_35.groupby(['Tender Round','EFA_block']).price.mean()
#TR_Price_75
#TR_Price_75.to_csv('TR_Price_75.csv')

### Median

TR_Price_median = df19.groupby(['Tender Round', 'EFA_block']).price.median()
TR_Price_median =  TR_Price_median.to_frame()
TR_Price_median.reset_index(inplace = True)
TR_Price_median = TR_Price_median.groupby(['Tender Round','EFA_block']).price.mean()
# TR_Price_median =  TR_Price_median.to_frame()
# TR_Price_median.reset_index(inplace = True)
#TR_Price_median
#TR_Price_median.to_csv('TR_Price_median.csv')

### Max Price

TR_Price_max = df19.groupby(['Tender Round', 'EFA_block']).price.max()
TR_Price_max =  TR_Price_max.to_frame()
TR_Price_max.reset_index(inplace = True)
TR_Price_max = TR_Price_max.groupby(['Tender Round', 'EFA_block']).price.mean()
# TR_Price_max.reset_index(inplace = True)
# TR_Price_max = TR_Price_max.groupby(['Tender Round']).price.mean()

# TR_Price_max.to_csv('Maximum.csv')

### 75th quantile

TR_Price_75 = df19.groupby(['Tender Round', 'EFA_block']).price.quantile(.75)
TR_Price_75 =  TR_Price_75.to_frame()
TR_Price_75.reset_index(inplace = True)
TR_Price_75 = TR_Price_75.groupby(['Tender Round','EFA_block']).price.mean()
#TR_Price_75
#TR_Price_75.to_csv('TR_Price_75.csv')

TR_Price_75

#TR_Price_75_R = TR_Price_75.drop(['Tender Round'], axis = 1)

### 95th quantile

TR_Price_95 = df19.groupby(['Tender Round', 'EFA_block']).price.quantile(.95)
TR_Price_95 =  TR_Price_95.to_frame()
TR_Price_95.reset_index(inplace = True)
TR_Price_95 = TR_Price_95.groupby(['Tender Round','EFA_block']).price.mean()
# TR_Price_95 =  TR_Price_95.to_frame()
# TR_Price_95.reset_index(inplace = True)
#TR_Price_95
#TR_Price_95.to_csv('TR_Price_95.csv')

#TR_Price_95_R = TR_Price_95.drop(['Tender Round'], axis = 1)

df20_grouped = df20.groupby(['Tender Round', 'EFA_block','End Date']).price.median()
df20_grouped =  df20_grouped.to_frame()
df20_grouped.reset_index(inplace = True)
df20_grouped["Months"] = df20_grouped['End Date'].dt.month
# TR_Price_median = TR_Price_median.groupby(['Tender Round']).price.mean()

df20_grouped

#df20_grouped1 = df20_grouped.groupby(['Tender Round', 'Months']).price.mean
# df20_grouped =  df20_grouped.to_frame()

## Multiple Variable column

data2["Months"] = data2['Month'].dt.month

data2_grouped = data2.groupby(['Tender Round','EFA_block','Months']).Average.mean()
data2_grouped =  data2_grouped.to_frame()
data2_grouped.reset_index(inplace = True)
# data2_grouped = data2_grouped[data2_grouped['Tender Round'] <= mats]

# data2_grouped = data2_grouped[data2_grouped['Tender Round'] >=112]
# data2_grouped = data2_grouped.reset_index(drop = True)

data2_grouped

dummy2 = sm.categorical(data2_grouped["Months"].values.reshape(1,-1), drop =True)
dummy2

dummy = sm.categorical(df20_grouped["Months"].values.reshape(1,-1), drop =True)
dummy

dummy3 = sm.categorical(df20_grouped["EFA_block"].values.reshape(1,-1), drop =True)
dummy3

dummy4 = sm.categorical(data2_grouped["EFA_block"].values.reshape(1,-1), drop =True)
dummy4

Average_Requirements.shape

median_variable.shape

dummy.shape

dummy3.shape

xData = np.hstack((dummy,dummy3, Average_Requirements,median_variable))

median_variable.shape

dummy2.shape

Future_Requirements.shape

variable_median1.shape

dummy4.shape

tData = np.hstack((dummy2,dummy4, Future_Requirements, variable_median1))

median_variable

model = sm.OLS(TR_Price_median, xData)
results = model.fit()
print(results.summary())



#print(results.summary())

# model = sm.OLS(TR_Price_median, tData)
# results = model.fit()
# print(results.summary())

tData.shape

xData.shape

safe = results.predict(tData)
print(safe)

## Regression analysis

lm = LinearRegression()
model_median = lm.fit(Average_Requirements,TR_Price_median)
forecast_median = model_median.predict(Future_Requirements)
forecast_median
#model_75 = lm.fit(Average_Requirements,TR_Price_75)
#model_95 = lm.fit(Average_Requirements,TR_Price_95)

model_median.score(Average_Requirements,TR_Price_median)

lm = LinearRegression()
model_75 = lm.fit(Average_Requirements,TR_Price_75)
forecast_75 = model_75.predict(Future_Requirements)
forecast_75

model_75.score(Average_Requirements,TR_Price_75)

lm = LinearRegression()
model_95 = lm.fit(Average_Requirements,TR_Price_95)
forecast_95 = model_95.predict(Future_Requirements)
forecast_95

model_95.score(Average_Requirements,TR_Price_95)

## Back Testing

X_train, x_test, Y_train, y_test = train_test_split(xData,TR_Price_median, test_size= 0.2)

linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

linear_model.score(X_train, Y_train)

linear_model.coef_

# predictors = X_train.columns
# coef = pd.Series(linear_model.coef_,predictors).sort_values()
# print(coef)

y_predict = linear_model.predict(x_test)

print(y_predict)

print(y_test)

mean_squared_error(y_test, y_predict)



r_square = linear_model.score(x_test,y_test)
r_square

fig = go.Figure()
fig.add_trace(go.Scatter(x=y_predict,y =y_test.values))
fig.show()

## Lasso Regression

lasso_model = Lasso(alpha=0.2, normalize=True)
lasso_model.fit(X_train,Y_train)

lasso_model.score(X_train,Y_train) 

# coef = pd.Series(lasso_model.coef_,predictors).sort_values()
# print(coef)

y_predict_l = lasso_model.predict(tData)

r_square = lasso_model.score(x_test,y_test)
r_square


print(y_predict)

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


TR_Price_median

median = median.reset_index(drop = True)
#median.drop(columns = 'Tender Round')
#median.reset_index(inplace = True)
#median

column = data2[['Tender Round', 'EFA_block','Month']]
# column = column.drop_duplicates(subset=['Tender Round'])
column = column.reset_index(drop=True)
#column

data2

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

# scenarios = scenarios.fillna(0)
# scenarios

scenarios['precise'] = scenarios['Month'].astype(str) + " " + scenarios['EFA_block']


scenarios.dtypes

scenarios1 = scenarios.tail(24)
# scenarios1 = scenarios

scenarios2 = scenarios.head(15)

scenarios1

scenarios1['Medium'] = scenarios1['Safe'] - 2

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

Title = 'FFR Dynamic EFA Block Model Predictions vs Actual'
fig = go.Figure()
fig.add_trace(go.Scatter(x=scenarios1["precise"],y =scenarios1["price"],
                    mode='markers+lines',
                    name='Average DFFR Accepted Price'))
# fig = px.scatter(scenarios2, x="Month", y="price")

fig.add_trace(go.Scatter(x=scenarios1["precise"], y =scenarios1["Max"],mode = 'lines+markers' ,name = "Maximum DFFR Accepted Price"))
fig.add_trace(go.Scatter(x=scenarios1["precise"], y =scenarios1["very_safe"],mode = 'lines+markers' ,name = "Safe Forecast"))
# fig.add_trace(go.Scatter(x=scenarios1["precise"], y =scenarios1["Safe"],mode = 'lines+markers' ,name = "Medium Forecast"))
fig.add_trace(go.Scatter(x=scenarios1["precise"], y =scenarios1["Medium"],mode = 'lines+markers' ,name = "Medium Forecast2"))

# fig.add_trace(go.Scatter(x=scenarios["EFA_block"], y =scenarios["Optimistic"],mode = 'lines+markers' ,name = "Aggressive Forecast"))
fig.update_layout(title = Title, xaxis_title = 'Month/EFA block', yaxis_title = 'Price (£)')
fig.layout.template = 'plotly_white'
fig.show()
plot(fig, filename = Title + '.html')
fig.write_image(Title + ".png")
# plot(fig)
scenarios3 = scenarios1.tail(12)
scenarios3.to_excel('DFFR Model 2 Predictions.xlsx')