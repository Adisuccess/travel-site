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
from pandas import ExcelWriter

def dynamic_accepted(x = None):
    data = pd.read_excel('Trial.xlsx')
    if x is not None:
        data = data.loc[data['Tender Round'].isin(x)]
    data = data[['Tender Date','Start Date', 'End Date', 'Reasoning','Tender Round', 'Tender Ref','Company Name','Tendered Unit','Status','Weekdays (From)', 'Weekdays (To)', 'Start EFA', 'End EFA','Static/Dynamic','Availability Fee (£/h)','Dynamic Primary (0.5Hz)', 'Dynamic Secondary (0.5Hz)', 'Dynamic High (0.5Hz)']]
    data = data[data["Static/Dynamic"] == 'Dynamic']
    data = data[data['Status'] == 'Accepted']
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

#newpath = r'/Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/Database/FFR'
#
#
#os.chdir(newpath)
##change working directory to save output file
#
##dfBH.to_excel("fafr.xlsx",index = False, header = None) # Saved google sheet to an excel file
#
#current_directory = os.getcwd()

#Edit the excel file
#mypath = current_directory + "/fafr.xlsx"


result = dynamic_accepted()

df19 = result
df19['Start EFA'] = (pd.to_numeric(df19['Start EFA'], errors='coerce')).fillna(0).astype(int)
df19['End EFA'] = (pd.to_numeric(df19['End EFA'], errors='coerce')).fillna(0).astype(int)
df19['blocks_nr']='test'
df19['blocks']='test'

result_s = static_accepted()

df20 = result_s
df20['Start EFA'] = (pd.to_numeric(df20['Start EFA'], errors='coerce')).fillna(0).astype(int)
df20['End EFA'] = (pd.to_numeric(df20['End EFA'], errors='coerce')).fillna(0).astype(int)
df20['blocks_nr']='test'
df20['blocks']='test'

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
#            print('element of the range: ', df20['blocks'].loc[i][j])
            df20['EFA_block'].loc[i] = 'EFA {}'.format(df20['blocks'].loc[i][j])
            j+=1
            
        elif(j==len(df20['blocks'].loc[i])):
            j=0
#            print('!!!!!!!!!: ', df20['blocks'].loc[i][j])
            df20['EFA_block'].loc[i] = 'EFA {}'.format(df20['blocks'].loc[i][j])
            j+=1
        
        elif(j>len(df20['blocks'].loc[i])):
            j=0
#            print('????????: ', df20['blocks'].loc[i][j])
            df20['EFA_block'].loc[i] = 'EFA {}'.format(df20['blocks'].loc[i][j])
            j+=1
        else:
            j=0


df20['price'] = df20['price'].astype(float)

#df19

## Graphing situation

df19

#df19_hg = pd.pivot_table(df19, values=['Tender Round','EFA_block'], index = None , aggfunc = np.mean ).reset_index()

df19 = df19[df19['Tender Round'] == df19['Tender Round'].max()]
df20 = df20[df20['Tender Round'] == df20['Tender Round'].max()]




### Median, 75th and 95th Quantile

TR_Price_median = df19.groupby(['Tender Round', 'EFA_block']).price.median()
TR_Price_median =  pd.DataFrame(TR_Price_median, columns=['price'])
TR_Price_median.reset_index(inplace = True)
# TR_Price_median = TR_Price_median.groupby(['Tender Round']).price.mean()

TR_Price_75 = df19.groupby(['Tender Round', 'EFA_block']).price.quantile(.75)
TR_Price_75 =  pd.DataFrame(TR_Price_75, columns= ['price'])
TR_Price_75.reset_index(inplace = True)
# TR_Price_75 = TR_Price_75.groupby(['Tender Round']).price.mean()

TR_Price_95 = df19.groupby(['Tender Round', 'EFA_block']).price.quantile(.95)
TR_Price_95 =  pd.DataFrame(TR_Price_95, columns= ['price'])
TR_Price_95.reset_index(inplace = True)
# TR_Price_95 = TR_Price_95.groupby(['Tender Round']).price.mean()

TR_Price_S = df20.groupby(['Tender Round', 'EFA_block']).price.median()
TR_Price_S =  pd.DataFrame(TR_Price_S, columns= ['price'])
TR_Price_S.reset_index(inplace = True)

volume = df19.groupby(['Tender Round', 'EFA_block']).agg(volume = ('Dynamic Primary (0.5Hz)', 'sum'))
#volume =  pd.DataFrame(volume, columns= ['volume'])
volume.reset_index(inplace = True)


# volume =  pd.DataFrame(volume, columns= ['Dynamic Primary (0.5Hz)'])
# volume_median.reset_index(inplace = True)
# volume_median = df19.groupby(['Tender Round', 'EFA_block']).agg(volume = ('Dynamic Primary (0.5Hz)', 'sum'))
volume

data2 = pd.read_excel('Requirements.xlsx')
data2 = data2.drop(['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 9', 'Unnamed: 8'], axis = 1)
data2['Sum_requirement'] = data2['Primary'] + data2['Secondary'] + data2['High']
data2['Average'] = (data2['Primary'] + data2['Secondary'] + data2['High'])/3

column2 = data2[['Tender Round', 'EFA_block']]
column2 = column2[column2['Tender Round'] == df19['Tender Round'].max()]
column2 = column2.drop_duplicates(subset=['Tender Round'])
column2 = column2.reset_index(drop=True)

frays =  [TR_Price_median, TR_Price_75, TR_Price_95, TR_Price_S,volume]


scenarios = pd.concat(frays, axis =1)
scenarios.columns = ['Tender Round', 'EFA Block', 'Price','Tender Round', 'EFA Block', '75th Quantile','Tender Round', 'EFA Block', '95th Quantile','Tender Round', 'EFA Block', 'Static Price','Tender Round', 'EFA Block', 'volume']
scenarios = scenarios.loc[:,~scenarios.columns.duplicated()]

#scenarios = scenarios.loc[:,~scenarios.columns.duplicated()]
# scenarios = scenarios.dropna()

# scenarios = scenarios[scenarios['Tender Round'] == scenarios_EFA['Tender Round'].max()]
# scenarios_EFA['Tender Round'] = 123

#print('Scenarios for TR 123')
#scenarios

#scenarios.columns = ['Median', '75th Quantile', '95th Quantile']
#scenarios['EFA Block'] = 
#scenarios

scenarios['Median Revenue'] = scenarios['Price'] * 4 * 7
scenarios['75th Revenue'] = scenarios['75th Quantile'] * 4 * 7
scenarios['95th Revenue'] = scenarios['95th Quantile'] * 4 * 7
scenarios['Static Revenue'] = scenarios['Static Price'] * 4 * 7


newpath = r'/Users/adieleejiofor/Dropbox (Kiwi Power Ltd)/27# Optimisation/02 ANALYSIS/01 Anciliary Market/Database/Fast Acting'


os.chdir(newpath)
## Fast Acting Data

#### LFS

LFS = pd.read_excel('Results Analysis_A.xlsx', sheet_name = 'Results Summary')

LFS = LFS[LFS['Service'] == 'LFS']
LFS = LFS[LFS['Week TR'] == LFS['Week TR'].max()]
#LFS

LFS_grouped = LFS.groupby(['EFA Blocks']).agg(LFS_Price=('Clearing Price', 'mean'), LFS_Vol = ('Cleared Volume', 'mean')).reset_index().reset_index()
LFS_grouped['LFS Revenue'] = LFS_grouped['LFS_Price'] * 28 

data = pd.read_excel('Results Analysis_A.xlsx', sheet_name = "Results by Unit")
data1 = pd.read_excel('Results Analysis_A.xlsx')

#data

data_l = data[data["Service_U"] == "LFS"]
data_l = data_l[data_l["Week TR_U"] == data["Week TR_U"].max()]
#data_l

data1_l = data1[data1["Service"] == "LFS"]
data1_l = data1_l[data1_l["Week TR"] == data1["Week TR"].max()]
#data1_l

data1_l_grouped = data1_l.groupby(['Week TR', 'EFA Blocks']).agg(avg_price=('Clearing Price', 'mean')).reset_index()

data_l_grouped = data_l.groupby(['Company','Week TR_U', 'EFA Blocks_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()

data_l_grouped['Avg_vol'] = round(data_l_grouped['Tot_vol']/7)
data_l_grouped["EFA Blocks"] = data_l_grouped["EFA Blocks_U"]

datar = data_l_grouped.merge(data1_l_grouped, on="EFA Blocks", how = 'inner'  )

datar.head()

data_l_grouped.head(5)

data1_l_grouped.head(5)


# fig = px.bar(datar, x="EFA Blocks", y="Avg_vol", color='Company')
# fig.add_trace(px.scatter(datar, y = "avg_price"))
# # fig = make_subplots(specs=[[{"secondary_y": True}]])
# # fig.add_trace(go.Scatter(y=datar["avg_price"]), secondary_y=True)
# fig.show()

### DLH 

DLH = pd.read_excel('Results Analysis_A.xlsx', sheet_name = 'Results Summary')

DLH = DLH[DLH['Service'] == 'DLH']
DLH = DLH[DLH['Week TR'] == DLH['Week TR'].max()]
#DLH

fig = px.scatter(DLH, x = 'Day ', y = 'Clearing Price', size = 'Cleared Volume', color = 'EFA Blocks')
fig.show()

data_2 = data[data["Service_U"] == "DLH"]
data_2 = data_2[data_2["Week TR_U"] == data_2["Week TR_U"].max()]
#data_l

data1_d = data1[data1["Service"] == "DLH"]
data1_d = data1_d[data1_d["Week TR"] == data1_d["Week TR"].max()]
#data2_l

data1_d_grouped = data1_d.groupby(['Week TR', 'EFA Blocks']).agg(avg_price=('Clearing Price', 'mean')).reset_index()

data_2_grouped = data_2.groupby(['Company','Week TR_U', 'EFA Blocks_U']).agg(Tot_vol=('Cleared Volume_U', 'sum')).reset_index()

data_2_grouped['Avg_vol'] = round(data_2_grouped['Tot_vol']/7)
data_2_grouped["EFA Blocks"] = data_2_grouped["EFA Blocks_U"]

datar2 = data_2_grouped.merge(data1_d_grouped, on="EFA Blocks", how = 'inner'  )

data_2_p = data_2.groupby(['EFA_U', 'Day_U', 'EFA Blocks_U']).agg(Price =('Clearing Price_U', 'mean'), Volume = ("Cleared Volume_U", 'sum')).reset_index()

DLH_grouped = DLH.groupby(['EFA Blocks']).agg(DLH_Price=('Clearing Price', 'mean'), DLH_Vol = ('Cleared Volume', 'mean')).reset_index()
DLH_grouped['DLH Revenue'] = DLH_grouped['DLH_Price'] * 28 

## Static FFR

#Title = Tender + ' Dynamic Market Results & Volumes'
#law = 'volume = ' + scenarios['volume'] 
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_scatter (x =scenarios["EFA Block"], y =scenarios["Price"], mode = 'markers', hovertext ='volume = '+scenarios['volume'].astype(str) , name = 'DFFR Median Price', marker =dict(size = scenarios['volume'] * 0.4))
fig.add_scatter (x =DLH_grouped["EFA Blocks"], y =DLH_grouped["DLH_Price"], mode = 'markers', hovertext ='volume = '+DLH_grouped['DLH_Vol'].astype(str) , name = 'DLH Price', marker =dict(size = DLH_grouped['DLH_Vol'] * 0.4))
fig.add_scatter (x =LFS_grouped["EFA Blocks"], y =LFS_grouped["LFS_Price"], mode = 'markers', hovertext ='volume = '+LFS_grouped['LFS_Vol'].astype(str) , name = 'LFS Price', marker =dict(size = LFS_grouped['LFS_Vol'] * 0.4))
fig.add_bar(x=scenarios["EFA Block"], y =scenarios["Median Revenue"], name = "FFR Average Revenue",secondary_y = True, marker =dict(opacity = 0.6)) 
            #secondary_y = True)

#fig.add_scatter(x=scenarios["EFA Block"], y =scenarios['75th Quantile'], mode = 'markers', name = '75th Quartile', hovertext = scenarios['volume'], marker = dict(size = scenarios['volume'] * 0.5))
#fig.add_bar(x=scenarios["EFA Block"], y =scenarios["75th Revenue"], name = "FFR 75th quantile Revenue",secondary_y = True, marker =dict(opacity = 0.8))
fig.add_bar(x=scenarios["EFA Block"], y =scenarios["95th Revenue"], name = "FFR Strategic Revenue",secondary_y = True, marker =dict(opacity = 0.6))
fig.add_bar(x=DLH_grouped["EFA Blocks"], y =DLH_grouped["DLH Revenue"], name = "DLH Average Revenue",secondary_y = True, marker =dict(opacity = 0.6))
#fig.add_scatter(x=scenarios["EFA Block"], y =scenarios['95th Quantile'], mode = 'markers', name = '95th Quartile', hovertext = scenarios['volume'], marker = dict(size = scenarios['volume'] * 0.5))
fig.add_bar(x=LFS_grouped["EFA Blocks"], y =LFS_grouped["LFS Revenue"], name = "LFS Average Revenue",secondary_y = True, marker =dict(opacity = 0.6))

fig.update_layout(showlegend = True,
    spikedistance =  -1,
)

fig.update_layout(title = 'Weekly Revenue Comparison', xaxis_title = 'EFA Block', yaxis_title = 'Price (£)')
fig.update_yaxes(title_text="Weekly Revenue (£)", secondary_y=True)
fig.layout.template = 'plotly_white'
fig.show()
# plot(fig)

new = pd.read_excel('Market Revenues.xlsx')
new1 = pd.read_excel('Market Revenues.xlsx', sheet_name = 'Day-ahead')
new2 = pd.read_excel('Market Revenues.xlsx', sheet_name = 'Imbalance')
new3 = pd.read_excel('Market Revenues.xlsx', sheet_name = 'Balancing Mechanism')

#Title = Tender + ' Dynamic Market Results & Volumes'
#law = 'volume = ' + scenarios['volume'] 
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_scatter (x =scenarios["EFA Block"], y =scenarios["Price"], mode = 'markers+lines', hovertext ='volume = '+scenarios['volume'].astype(str) , name = 'DFFR Median Price', marker =dict(color='#77FFCD'))
fig.add_scatter (x =scenarios["EFA Block"], y =scenarios["Static Price"], mode = 'markers+lines', hovertext ='volume = '+scenarios['volume'].astype(str) , name = 'SFFR Median Price', marker =dict(color='#203139'))                 
fig.add_scatter (x =DLH_grouped["EFA Blocks"], y =DLH_grouped["DLH_Price"], mode = 'markers+lines', hovertext ='volume = '+DLH_grouped['DLH_Vol'].astype(str) , name = 'DLH Price', marker =dict(color='#D13A6F'))
fig.add_scatter (x =LFS_grouped["EFA Blocks"], y =LFS_grouped["LFS_Price"], mode = 'markers+lines', hovertext ='volume = '+LFS_grouped['LFS_Vol'].astype(str) , name = 'LFS Price', marker =dict(color='#3ABAD1'))
fig.add_scatter (x =new["efa_block"], y =new["avgPrice"], mode = 'markers+lines', name = 'Intraday Average', marker =dict(color='#3870D1'))
fig.add_scatter (x =new1["efa_block"], y =new1["avgPrice"], mode = 'markers+lines', name = 'Day Ahead Average', marker =dict(color='#9B3AD2'))
fig.add_scatter (x =new2["efa_block"], y =new2["avgPrice"], mode = 'markers+lines', name = 'Imbalance Average', marker =dict(color='#BCC3C6'))
fig.add_scatter (x =new3["efa_block"], y =new3["avgPrice"], mode = 'markers+lines', name = 'Balancing Mechanism Average', marker =dict(color='#00A06E'))

 
fig.add_bar(x=scenarios["EFA Block"], y =scenarios["Median Revenue"], name = "DFFR Average Revenue",secondary_y = True, marker =dict(opacity = 0.6, color='#77FFCD')) 
fig.add_bar(x=scenarios["EFA Block"], y =scenarios["Static Revenue"], name = "SFFR Average Revenue",secondary_y = True, marker =dict(opacity = 0.6, color='#203139')) 
# fig.add_bar(x=scenarios["EFA Block"], y =scenarios["95th Revenue"], name = "FFR Strategic Revenue",secondary_y = True, marker =dict(opacity = 0.6))
fig.add_bar(x=DLH_grouped["EFA Blocks"], y =DLH_grouped["DLH Revenue"], name = "DLH Average Revenue",secondary_y = True, marker =dict(opacity = 0.6,color='#D13A6F'))
fig.add_bar(x=LFS_grouped["EFA Blocks"], y =LFS_grouped["LFS Revenue"], name = "LFS Average Revenue",secondary_y = True, marker =dict(opacity = 0.6,color='#3ABAD1'))
fig.add_bar(x=new["efa_block"], y = new["Revenues"], name = "Intraday Revenue",secondary_y = True, marker =dict(opacity = 0.6,color='#3870D1'))
fig.add_bar(x=new1["efa_block"], y = new1["Revenues"], name = "Day Ahead Revenue",secondary_y = True, marker =dict(opacity = 0.6,color='#9B3AD2'))
fig.add_bar(x=new2["efa_block"], y = new2["Revenues"], name = "Imbalance Revenue",secondary_y = True, marker =dict(opacity = 0.6,color='#BCC3C6'))
fig.add_bar(x=new3["efa_block"], y = new3["Revenues_1MW"], name = "Balancing Mechanism Revenue",secondary_y = True, marker =dict(opacity = 0.6,color='#00A06E'))


fig.update_layout(showlegend = True,
    spikedistance =  -1,
)

fig.update_layout(title = '<b>Weekly Revenue Comparison</b>', xaxis_title = '<b>EFA Block</b>', yaxis_title = '<b>Price (£)</b>')
fig.update_yaxes(title_text="Weekly Revenue (£)", secondary_y=True)
fig.layout.template = 'plotly_white'
fig.show()
plot(fig)

scenarios['Annualized Revenue'] = scenarios['Median Revenue'] * 52
DLH_grouped['Annualized Revenue'] = DLH_grouped['DLH Revenue'] * 52
LFS_grouped['Annualized Revenue'] = LFS_grouped['LFS Revenue'] * 52

writer = ExcelWriter('Ancillary Markets.xlsx')
scenarios.to_excel(writer, 'Dynamic FFR')
DLH_grouped.to_excel(writer, 'Fast Acting DLH')
LFS_grouped.to_excel(writer, 'Fast Acting LFS')


writer.save()