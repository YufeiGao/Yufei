#%%
"""
@Project: alpha101

@FileName: Trials on alpha101

@Author：Yufei Gao

@Create date: 2018/7/27

@description：do some regression on the sample alphas

@Update date：  

@Vindicator：  

"""  
#%%
import os
os.chdir('D:\\gyf\\alpha101')
import pandas as pd
from alpha_func import *
from signal_Test import *
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import multiprocessing
import matplotlib.pyplot as plt
import scipy.io
#%%
province_file = pd.read_csv('C://Users/HP/Desktop/stock_province.csv',encoding='gbk');
data_province = pd.DataFrame(province_file);
data_province.fillna(0, inplace=True);
data_province.set_index(['code'],inplace=True);
#%%
import datetime
def no_before_endddate(enddate, location):
    end_pos = enddate.split('/');
    end_time = datetime.datetime(int(end_pos[0]),int(end_pos[1]),int(end_pos[2]));
    no_in_province = 0;
    for key in data_province.index:
        ipo_pos = data_province['ipo_date'][key].split('-');
        ipo_time = datetime.datetime(int(ipo_pos[0]),int(ipo_pos[1]),int(ipo_pos[2]));
        if (ipo_time < end_time) & (data_province['province'][key] == location):
            no_in_province = no_in_province + 1;
    return no_in_province;
#%%
# get the province list
temp = [];
for key in data_province.index:
    temp.append(data_province['province'][key]);
province_list = list(set(temp))
#%%
num_list = [];
for item in province_list:
    num_list.append(no_before_endddate('2018/1/1', item));
num_series = pd.Series(num_list, index = province_list);            
#%%
def single_stock_pct(key, startdate, enddate):
    # find start position
    temp_pos = startdate.split('/');
    temp_time = datetime.datetime(int(temp_pos[0]),int(temp_pos[1]),int(temp_pos[2]));
    start_pos = -1;
    for day in data_pct.index:
        start_pos += 1;
        cur_pos = day.split('/');
        cur_time = datetime.datetime(int(cur_pos[0]),int(cur_pos[1]),int(cur_pos[2]));
        if (cur_time >= temp_time):
            break;
    
    # find end position
    temp_pos = enddate.split('/');
    temp_time = datetime.datetime(int(temp_pos[0]),int(temp_pos[1]),int(temp_pos[2]));
    end_pos = -1;
    for day in data_pct.index:
        end_pos += 1;
        cur_pos = day.split('/');
        cur_time = datetime.datetime(int(cur_pos[0]),int(cur_pos[1]),int(cur_pos[2]));
        if (cur_time > temp_time):
            end_pos = end_pos - 1;
            break;
            
    #cpt pct over the period
    cum_pct = 1.0;
    for day in range(start_pos, end_pos+1):
        cur_pct = get_pct(day, pd.Series({key:0.0}));
        cum_pct = cum_pct*(1+cur_pct[0]/100.0);
    return (cum_pct - 1)*100.0;
#%%
def stock_pct_province(startdate, enddate, location):
    temp_pos = startdate.split('/');
    temp_time = datetime.datetime(int(temp_pos[0]),int(temp_pos[1]),int(temp_pos[2]));
    stock_pct_list = [];
    for key in data_province.index:
        if (data_province['province'][key] == location):
            cur_ipo = data_province['ipo_date'][key];
            cur_pos = cur_ipo.split('-');
            ipo_time = datetime.datetime(int(cur_pos[0]),int(cur_pos[1]),int(cur_pos[2]));
            dlt = datetime.timedelta(days = 90);  #剔除ipo不足90天的股票
            if(temp_time - dlt > ipo_time):           
                cur_pct = single_stock_pct(key, startdate, enddate);
                stock_pct_list.append(cur_pct);
    final_list = list(filter(lambda x: (x!=0.0), stock_pct_list)); #剔除停牌股
    return np.nanmean(final_list);
#%%
temp = [];
for key in province_list:
    temp.append(2*stock_pct_province('2018/1/1', '2018/6/30', key));
pct_province_series = pd.Series(temp, index = province_list);
    
#%%
def stock_pct_province_and_industry(startdate, enddate, location, ind):
    temp_pos = startdate.split('/');
    temp_time = datetime.datetime(int(temp_pos[0]),int(temp_pos[1]),int(temp_pos[2]));
    stock_pct_list = [];
    for key in data_province.index:
        if ((data_province['province'][key] == location)&(data_province['industry'][key] == ind)):
            cur_ipo = data_province['ipo_date'][key];
            cur_pos = cur_ipo.split('-');
            ipo_time = datetime.datetime(int(cur_pos[0]),int(cur_pos[1]),int(cur_pos[2]));
            dlt = datetime.timedelta(days = 90);  #剔除ipo不足90天的股票
            if(temp_time - dlt > ipo_time):           
                cur_pct = single_stock_pct(key, startdate, enddate);
                stock_pct_list.append(cur_pct);
    final_list = list(filter(lambda x: (x!=0.0), stock_pct_list)); #剔除停牌股
    return np.nanmean(final_list);
#%%
# get the province list
temp = [];
for key in data_province.index:
    temp.append(data_province['industry'][key]);
industry_list = list(set(temp))
#%%
startdate = '2010/1/1';
enddate = '2018/6/30';
name_list = [[] for i in range(5)];
pct_list = [[] for i in range(5)];
for province in province_list:
    pct_by_ind = [stock_pct_province_and_industry(startdate, enddate, province, ind) for ind in industry_list];
    pct_series = pd.Series(pct_by_ind, index = industry_list);
    pct_series.sort_values(ascending = False, inplace = True);
    for i in range(5):
        name_list[i].append(pct_series.index[i]);
        pct_list[i].append(pct_series[i]);
#%%
startdate = '2010/1/1';
enddate = '2011/1/1';
province = '宁夏回族自治区';
pct_by_ind = [stock_pct_province_and_industry(startdate, enddate, province, ind) for ind in industry_list];
pct_series = pd.Series(pct_by_ind, index = industry_list);
pct_series.sort_values(ascending = False, inplace = True);        
    
#%%
import numpy.matlib 
import numpy as np 
print (np.matlib.empty((2,2)))  

    
    
        

