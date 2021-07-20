# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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


## Refer to the Tender round you want.

#x = [121]
result = dynamic_accepted()

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


#df19

## Graphing situation

df19

#df19_hg = pd.pivot_table(df19, values=['Tender Round','EFA_block'], index = None , aggfunc = np.mean ).reset_index()


#df19_hg

df19_grouped = df19.groupby(['Tender Round','EFA_block']).agg(avg_price=('price', 'median'), total_volume=('Dynamic Secondary (0.5Hz)', 'sum')).reset_index()

df19

df19_grouped

fig = px.scatter(df19_grouped, x ="Tender Round", y ="avg_price", size = "total_volume", color= "EFA_block")
fig.show()

df19_grouped2 = df19[df19['EFA_block'] == 'EFA 5' ]

df19_grouped2

# df = px.data.tips()
fig = px.box(df19_grouped2, x="Tender Round", y = "price")
fig.show()

# Price = go.Box(
#     x = df19['Tender Round'],
#     y = df19[['price']].median(axis =1),
# )

# Price1 = go.Scatter(
#     x = df19['EFA_block'],
#     y = df19['price'],
#     mode = 'markers+lines'
# )

df19

# df = px.data.tips()
fig = px.box(df19, x="Tender Round", y = "price")
fig.show()

## Break it down into EFA blocks



## Getting the Requirements 

data2 = pd.read_excel('Requirements.xlsx')
data2 = data2.drop(['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 9', 'Unnamed: 8'], axis = 1)
data2['Sum_requirement'] = data2['Primary'] + data2['Secondary'] + data2['High']
data2['Average'] = (data2['Primary'] + data2['Secondary'] + data2['High'])/3

## Fit Requirements to EFA rounds

mat = df19['Tender Round'].max()
data3 = data2[data2['Tender Round'] <= mat]
Average_Requirements = data3.groupby(['Tender Round']).Average.mean().values.reshape(-1,1)
Future_Requirements = data2.groupby(['Tender Round']).Average.mean().values.reshape(-1,1)
#Average_Requirements

## Scenario Forecasts (median , 75th and 95th)
### Median

TR_Price_median = df19.groupby(['Tender Round', 'EFA_block']).price.median()
TR_Price_median =  TR_Price_median.to_frame()
TR_Price_median.reset_index(inplace = True)
TR_Price_median = TR_Price_median.groupby(['Tender Round']).price.mean()
# TR_Price_median =  TR_Price_median.to_frame()
# TR_Price_median.reset_index(inplace = True)
#TR_Price_median
#TR_Price_median.to_csv('TR_Price_median.csv')

#TR_Price_Median_R = TR_Price_median.drop(['Tender Round'], axis = 1)

### 75th quantile

TR_Price_75 = df19.groupby(['Tender Round', 'EFA_block']).price.quantile(.75)
TR_Price_75 =  TR_Price_75.to_frame()
TR_Price_75.reset_index(inplace = True)
TR_Price_75 = TR_Price_75.groupby(['Tender Round']).price.mean()
#TR_Price_75
#TR_Price_75.to_csv('TR_Price_75.csv')

TR_Price_75.head(20)

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

## Regression analysis

Future_Requirements.shape

TR_Price_median.shape

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

## Add results to Dataframe to create scenarios

median = pd.DataFrame(TR_Price_median, columns=['price'])
safe = pd.DataFrame(forecast_median, columns=['Safe'])
medium = pd.DataFrame(forecast_75, columns=['Medium'])
high = pd.DataFrame(forecast_95, columns=['High'])


median = median.reset_index(drop = True)
#median.drop(columns = 'Tender Round')
#median.reset_index(inplace = True)
#median

column = data2[['Tender Round', 'Month']]
column = column.drop_duplicates(subset=['Tender Round'])
column = column.reset_index(drop=True)
#column

frays = [column, median, safe, medium, high ]

scenarios = pd.concat(frays, axis = 1)
scenarios

## Scenarios from Last TR

safe_TR = pd.DataFrame(TR_Price_median, columns=['price'])
medium_TR = pd.DataFrame(TR_Price_75, columns=['price'])
high_TR = pd.DataFrame(TR_Price_95, columns=['price'])



safe_TR = safe_TR.reset_index(drop = True)
medium_TR = medium_TR.reset_index(drop = True)
high_TR = high_TR.reset_index(drop = True)

column1 = data2[['Tender Round', 'Month']]
column1 = column1.drop_duplicates(subset=['Tender Round'])
column1 = column1.reset_index(drop=True)

frys = [column1, safe_TR, medium_TR, high_TR ]

scenarios_TR = pd.concat(frys, axis = 1)
scenarios_TR.columns = ['Tender Round', 'Month', 'Safe', 'Medium', 'High']
scenarios_TR = scenarios_TR.dropna()
scenarios_TR

## Broken Down by EFA 

TR_Price_median_EFA = df19.groupby(['Tender Round', 'EFA_block']).price.median()
TR_Price_75_EFA = df19.groupby(['Tender Round', 'EFA_block']).price.quantile(.75)
TR_Price_95_EFA = df19.groupby(['Tender Round', 'EFA_block']).price.quantile(.95)

safe_EFA = pd.DataFrame(TR_Price_median_EFA, columns=['price'])
medium_EFA = pd.DataFrame(TR_Price_75_EFA, columns=['price'])
high_EFA = pd.DataFrame(TR_Price_95_EFA, columns=['price'])


safe_EFA = safe_EFA.reset_index(drop = True)
medium_EFA = medium_EFA.reset_index(drop = True)
high_EFA = high_EFA.reset_index(drop = True)

column2 = data2[['Tender Round', 'EFA_block']]
# column2 = column2.drop_duplicates(subset=['EFA_block'])
# column2 = column2.reset_index(drop=True)

flys = [column2, safe_EFA, medium_EFA, high_EFA ]

#data2

scenarios_EFA = pd.concat(flys, axis = 1)
scenarios_EFA.columns = ['Tender Round', 'EFA Block', 'Safe', 'Medium', 'High']
scenarios_EFA = scenarios_EFA.dropna()

scenarios_EFA = scenarios_EFA[scenarios_EFA['Tender Round'] == scenarios_EFA['Tender Round'].max()]
scenarios_EFA['Tender Round'] = 123

print('Scenarios for TR 123')
scenarios_EFA


# data2.tail(36)

#scenarios.to_csv('Scenarios.csv')

## Graph Scenarios

# scenarios_hg = pd.pivot_table(scenarios, values=['Safe', 'Medium', 'High'], index = 'Month')

# scenarios_hg

fig = px.line(scenarios, x = 'Month',  y='Safe')

# Only thing I figured is - I could do this 
fig.add_scatter(x =scenarios['Month'] , y=scenarios['Medium']) # Not what is desired - need a line 
fig.add_scatter(x =scenarios['Month'] , y=scenarios['High'])
#fig.add_scatter(x =scenarios['Month'] , y=scenarios['price'])

# Show plot 
fig.show()

scenarios_hist = scenarios[['Safe','Medium', 'High']]
scenarios_hist.hist(grid = False, color = "cadetblue")

## Calculate excess kurtosis

safe_kurtosis = kurtosis(scenarios['Safe'], fisher = True)
medium_kurtosis = kurtosis(scenarios['Medium'], fisher = True)
high_kurtosis = kurtosis(scenarios['High'], fisher = True)


safe_skew = skew(scenarios['Safe'])
medium_skew = skew(scenarios['Medium'])
high_skew = skew(scenarios['High'])

display("safe_kurtosis: {:.2}".format(safe_kurtosis))
display("medium_kurtosis: {:.2}".format(medium_kurtosis))
display("high_kurtosis: {:.2}".format(high_kurtosis))

display("safe_skew: {:.2}".format(safe_skew))
display("medium_skew: {:.2}".format(medium_skew))
display("medium_skew: {:.2}".format(medium_skew))

display("Safe")
display(stats.kurtosistest(scenarios['Safe']))

## Split month ahead predictions into EFA using percentage from last TR

## Plot market size and price on dash graph

# scenarios1 = scenarios.head(22)
# scenarios1

Forecast = go.Scatter(
    x = scenarios['Month'],
    y = scenarios['Safe'],
    mode = 'markers+lines',
    name = 'Forecast'
)

Medium_Risk = go.Scatter(
    x = scenarios['Month'],
    y = scenarios['Medium'],
    mode = 'markers+lines',
    name = 'Medium Scenario'
)

High_Risk = go.Scatter(
    x = scenarios['Month'],
    y = scenarios['High'],
    mode = 'markers+lines',
    name = 'High Risk'
)

Actual_Price = go.Scatter(
    x = scenarios['Month'],
    y = scenarios['price'],
    mode = 'markers+lines',
    name = 'Actual Price'
)

Tender = 'Tender Round ' + str(df19['Tender Round'].max())
Title = 'FFR Dynamic Model Predictions vs Actual'
fig = go.Figure()
fig.add_trace(go.Scatter(x=scenarios["Month"],y =scenarios["price"],
                    mode='lines+markers',
                    name='Actual Price'))
fig.add_trace(go.Scatter(x=scenarios["Month"], y =scenarios["Safe"],mode = 'lines+markers' ,name = "Forecast"))
fig.update_layout(title = Title, xaxis_title = 'Month', yaxis_title = 'Price (£)')
fig.layout.template = 'plotly_dark'
fig.show()
plot(fig)

## Static FFR

def static_accepted(filename):
    data = filename[['Tender Date', 'End Date','Tender Round', 'Tender Ref','Company Name','Tendered Unit','Status','Weekdays (From)', 'Weekdays (To)', 'Start EFA', 'End EFA','Static/Dynamic','Availability Fee (£/h)','Static Secondary']]
    data = data[data["Static/Dynamic"] == 'Static']
    result = data[data['Status'] == 'Accepted']

    #result_filtered = result[(result['Dynamic Primary (0.5Hz)'] != result['Dynamic High (0.5Hz)']) & (result['Dynamic Secondary (0.5Hz)'] != result['Dynamic High (0.5Hz)'])]
    result['price'] = result['Availability Fee (£/h)']/result['Static Secondary']
    #result = result[result['price'] != 0]
    result['Time_dif'] = result['End Date'] - result['Tender Date']
    result['Time_dif'] = pd.to_numeric(result['Time_dif'].dt.days, downcast = 'integer')
    result = result[result['Time_dif'] <= 62]
    return result

data = pd.read_excel('Trial.xlsx')
result_S = static_accepted(data)
result_S

df20 = result_S
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

EFA_Price_S = df20.groupby(['Tender Round', 'EFA_block']).price.mean()

EFA_Price_S

st = EFA_Price_S.to_frame()
st.reset_index(inplace = True)
st.head(10)

Average_Price_EFA_S = st.groupby(['Tender Round']).price.mean()
Average_Price_EFA_S

#st.to_csv('static.csv')

# Static Requirements 

data_s = pd.read_excel('Requirements.xlsx', sheet_name = 'Static')

data_s

data_s_1 = data_s[data_s['Tender Round'] <= mat]
Average_Requirements_s = data_s_1.groupby(['Tender Round']).Secondary.mean().values.reshape(-1,1)
Average_Requirements_s

st_r = st.groupby('Tender Round').price.mean()
st_r = st_r.to_frame()
st_r.reset_index(inplace = True)
st_r

st_r = st_r.drop(['Tender Round'], axis=1)
st_r

lm = LinearRegression()
model_st = lm.fit(Average_Requirements_s,st_r)

model_st.score(Average_Requirements_s,st_r)

Future_Requirements_s = data_s.groupby(['Tender Round']).Secondary.mean().values.reshape(-1,1)

lst = model_st.predict(Future_Requirements_s)

safe_s = pd.DataFrame(lst, columns=['Safe'])


EFA_Price_y = df20.groupby(['Tender Round']).price.mean()

EFA_Price_y

colun = data2[['Tender Round', 'Month', 'EFA_block']]


colun

s_s = data_s.groupby(['Tender Round']).Secondary.mean()
s_s = pd.DataFrame(s_s, columns=['Secondary'])
s_s.reset_index(inplace = True)

safe_s['Tender Round'] = s_s['Tender Round']

column_s = data_s[['Tender Round', 'Month']]
column_s = column.drop_duplicates(subset=['Tender Round'])
column_s = column.reset_index(drop=True)
#column

scenarios_s = pd.merge(left=column_s, right=safe_s, left_on = 'Tender Round', right_on = 'Tender Round')
#scenarios_s = pd.merge(left=scenarios_s, right=median_s,left_on = 'Tender Round', right_on = 'Tender Round')

frais = (scenarios_s, median_s)
scenarios_s = pd.concat(frais, axis = 1)

scenarios_s = scenarios_s.loc[:,~scenarios_s.columns.duplicated()]



scenarios_s 

Title = 'FFR Static Model Predictions vs Actual'
fig = go.Figure()
fig.add_trace(go.Scatter(x=scenarios_s["Month"],y =scenarios_s["price"],
                    mode='lines+markers',
                    name='Actual Price'))
fig.add_trace(go.Scatter(x=scenarios_s["Month"], y =scenarios_s["Safe"],mode = 'lines+markers' ,name = "Forecast"))
fig.update_layout(title = Title, xaxis_title = 'Month', yaxis_title = 'Price (£)')
fig.layout.template = 'plotly_dark'
fig.show()
# plot(fig)

## Amber

scenarios_l = scenarios[scenarios['Tender Round'] <= mat]

Tender = 'Tender Round ' + str(df19['Tender Round'].max())
Title = 'FFR Dynamic Model Predictions vs Actual'
fig = go.Figure()
fig.add_trace(go.Scatter(x=scenarios_l["Month"],y =scenarios_l["price"],
                    mode='lines+markers',
                    name='Actual Price'))
fig.add_trace(go.Scatter(x=scenarios_l["Month"], y =scenarios_l["Safe"],mode = 'lines+markers' ,name = "Forecast"))
fig.update_layout(title = Title, xaxis_title = 'Month', yaxis_title = 'Price (£)')
fig.layout.template = 'plotly_dark'
fig.show()
# plot(fig)

## Capacity Graphs

def month_index(result):
    df20 = result
    df20['Months_list'] = 'adi'
    df20['nr_months'] = 'genius'
    df20['months'] = 'is'
    for i in range(0, len(df20)):
        start_date = df20['Start Date'].iloc[i]
        end_date = df20['End Date'].iloc[i]
        df20['Months_list'].iloc[i] = pd.date_range(start_date ,end_date, 
                  freq='MS').strftime("%b-%Y").tolist()
        df20['nr_months'].iloc[i] = len(df20['Months_list'].iloc[i])
    test1 = df20.reindex(df20.index.repeat(df20.nr_months)).reset_index()
    return test1

test1 = month_index(result)

#test1

month = 0
for i in range(1, len(test1)):
    if(test1['Tender Ref'].loc[i-1] == test1['Tender Ref'].loc[i]):
        if (month <= tes1t['nr_months'].loc[i-1]):
            test1['months'].loc[i-1] = test1['Months_list'].loc[i-1][month]
            month+=1
    else:
        test1['months'].loc[i-1] = test1['Months_list'].loc[i-1][month]
        month=0

df19 = test1
df19.dropna(inplace = True)

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


df19['EFA_block'] = 'adi'
j=0
for i in range(0,len(df19)):
    print ('==================', i)
    if (df19['blocks_nr'].iloc[i]==1):
        #print('mpikame sto prwto if')
        print('First IF condition\n')
        j=0
        #print(df19['Start EFA'].loc[i], df19['End EFA'].loc[i])
        df19['EFA_block'].iloc[i] = 'EFA {}'.format(df19['Start EFA'].iloc[i])
        #print('>>>>>>>>>>>>>>> ',df19['EFA_block'].loc[i])
        print('>>>>>>>>>>>>>>> our EFA block is: ',df19['EFA_block'].iloc[i])
        
    else:
#         print('mpikame sto', df19['blocks'].loc[i], ' to opoio exei length ', len(df19['blocks'].loc[i]))
#         print('poso einai to j? ', j)
        print('ELSE we are in ', df19['blocks'].iloc[i], ' and its length is ', len(df19['blocks'].iloc[i]))
        print('What is j: ', j)
        if(j<len(df19['blocks'].iloc[i])):
#             print('to j einai: ',j)
#             print('j is: ', j)
#             print(len(df19['blocks'].loc[i]),len(df19['blocks'].loc[i]))
            print('element of the range: ', df19['blocks'].iloc[i][j])
           
            df19['EFA_block'].iloc[i] = 'EFA {}'.format(df19['blocks'].iloc[i][j])
#             print('----------------------',df19['EFA_block'].loc[i])

            j+=1
            
        elif(j==len(df19['blocks'].iloc[i])):
            j=0
            print('!!!!!!!!!: ', df19['blocks'].iloc[i][j])
            df19['EFA_block'].iloc[i] = 'EFA {}'.format(df19['blocks'].iloc[i][j])
#             print('----------------------',df19['EFA_block'].loc[i])
            j+=1
        
        elif(j>len(df19['blocks'].iloc[i])):
            j=0
            print('????????: ', df19['blocks'].iloc[i][j])
            df19['EFA_block'].iloc[i] = 'EFA {}'.format(df19['blocks'].iloc[i][j])
#             print('----------------------',df19['EFA_block'].loc[i])
            j+=1
        else:
            j=0


Capacity_duplicates = df19.sort_values('Dynamic Secondary (0.5Hz)', ascending=False).drop_duplicates(['Tendered Unit', 'months'], keep='last').sort_index()
Capacity_accepted = Capacity_duplicates.groupby(['months'])['Dynamic Secondary (0.5Hz)'].sum()

Capacity_acc = Capacity_duplicates.groupby(['months', 'EFA_block']).agg(price = ('price', 'mean'), Dynamic_Secondary  = ('Dynamic Secondary (0.5Hz)', 'sum')).reset_index()

# Cap_acc = go.Bar(
#     x = Capacity_duplicates['months'],
#     #y = test1[['Dynamic Primary (0.5Hz)', 'Dynamic Secondary (0.5Hz)','Dynamic High (0.5Hz)']].mean(axis=1),
#     y = Capacity_duplicates[['Dynamic Secondary (0.5Hz)']].sum(axis=1),
#     name = 'Capacity'
# )

# Price = go.Scatter(
#     x = test1['months'],
#     y = test1[['price']].mean(axis=1),
#     name = 'Price',
#     mode = 'markers+lines'
# )

Capacity_acc

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Bar(x= Capacity_acc['months'], y=Capacity_acc['Dynamic_Secondary'] ,name="Capacity"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=Capacity_acc['months'], y=Capacity_acc['price'], name="Price", mode = 'markers'),
    secondary_y=True,
)

fig.update_layout(title_text = "Dynamic FFR")


fig.update_xaxes(title_text = 'Month')

# Set y-axes titles
fig.update_yaxes(title_text="Capacity", secondary_y=False)
fig.update_yaxes(title_text="Price (£/MWh)", secondary_y=True)

fig.show()

data = [Cap_acc, Price]
layout = go.Layout(barmode='group')

fig = go.Figure(data = data, layout = layout)
iplot(fig, filename = 'grouped-bar')

scenarios.to_csv('scenarios.csv')