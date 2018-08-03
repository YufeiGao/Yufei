#%%
"""
@Project: alpha101

@FileName: Trials on alpha101

@Author：Yufei Gao

@Create date: 2018/7/25

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
file4 = pd.read_csv('C://Users/HP/Desktop/pe_d.csv');
data_pe = pd.DataFrame(file4);
data_pe.fillna(0, inplace=True);
data_pe.set_index(['Date'],inplace=True);
#%%
pb_file = pd.read_csv('C://Users/HP/Desktop/Pb_d.csv');
data_pb = pd.DataFrame(pb_file);
data_pb.fillna(0, inplace=True);
data_pb.set_index(['Date'],inplace=True);
#%%
frees_file = pd.read_csv('C://Users/HP/Desktop/Frees_d.csv');
data_frees = pd.DataFrame(frees_file);
data_frees.fillna(0, inplace=True);
data_frees.set_index(['Date'],inplace=True);
#%%
def pe_factor(enddate):
    pe_list = [];
    enddate = enddate - (date_to_int('2005/1/4'));
    for key,value in data_close.iteritems():
        if (key not in data_pe.columns):
            pe_list.append(float('nan'));
        elif (data_pe[key][enddate] <= 0.0):
            pe_list.append(float('nan')); #delete negtive values
        else:
            pe_list.append(data_pe[key][enddate]);
    final_series = pd.Series(pe_list, index = stock_code);
    final_series.sort_values(ascending=True, inplace=True);
    return final_series; 
#%%     
def frees_norm(enddate):
    frees_list = [];
    for key,value in data_close.iteritems():
        if (key not in data_frees.columns):
            frees_list.append(float('nan'));
        else:
            frees_list.append(data_frees[key][enddate]);
    sigma = np.nanstd(frees_list); #cpt the standard deviation
    mu = np.nanmean(frees_list); #cpt the mean
    cap = mu + 1.5*sigma;  # set the cap
    norm_list = [];
    for item in frees_list:
        if (item > cap):
            norm_list.append(1.5);
        else:
            norm_list.append((item - mu) / sigma);
    final_series = pd.Series(norm_list, index = stock_code);
    return final_series; 
def pe_norm(enddate):
    drop_list = []; # stock with nagetive PE should be droped
    pe_list = [];
    for key,value in data_close.iteritems():
        if (key not in data_pe.columns):
            drop_list.append(key);
            pe_list.append(float('nan'));
        elif (data_pe[key][enddate] <= 0.0):
            drop_list.append(key);
            pe_list.append(float('nan'));
        else:
            pe_list.append(data_pe[key][enddate]);
    sigma = np.nanstd(pe_list);
    mu = np.nanmean(pe_list);
    norm_list = [];
    cap = mu + 1.5*sigma;
    for item in pe_list:
        if (item > cap):
            norm_list.append(1.5);
        else:
            norm_list.append((item - mu) / sigma);
    final_series = pd.Series(norm_list, index = stock_code);
    return final_series, drop_list;
#%% 
def get_reg_data(enddate):
    frees_series = frees_norm(enddate);
    pe_series, drop_list = pe_norm(enddate);
    frees_series.drop(drop_list, inplace = True);
    pe_series.drop(drop_list, inplace = True);
    frame = [frees_series, pe_series];
    df = pd.concat(frame, axis=1);
    df.columns = ['x','y'];
    return df;
#%%
from sklearn import linear_model
def res_factor(enddate):
    enddate = enddate - (date_to_int('2005/1/4'));
    res_list = [];
    df = get_reg_data(enddate);
    regr = linear_model.LinearRegression();
    regr.fit(df['x'].values.reshape(-1,1), df['y']);
    a, b = regr.coef_, regr.intercept_;
    stock_idx = df.index;
    for key in stock_idx:
        res = df['y'][key] - b*df['x'][key] - a;
        res_list.append(res[0]);
    final_series = pd.Series(res_list, index = stock_idx);
    final_series.sort_values(ascending=True, inplace=True);
    return final_series;    
#%%
def set_dummy():
    industry_list = list(data_ipo['industry_sw']);
    industry_set = set(industry_list);
    ind_list = list(industry_set);
    idx = list(data_ipo['industry_sw'].index);
    idx_series = pd.Series(idx, name = 'code');
    dummy_list = [[] for i in range(28)]; #set 28 dummy series
    dummy_series = [];
    for i in range(28):
        for key in range(len(data_ipo)):
            if (data_ipo['industry_sw'][key] == ind_list[i]):
                dummy_list[i].append(1);
            else:
                dummy_list[i].append(0);
        cur_series = pd.Series(dummy_list[i], name = ind_list[i]);
        dummy_series.append(cur_series);
    frame = [idx_series];
    for i in range(len(dummy_series)):
        frame.append(dummy_series[i]);
    dummy_df = pd.concat(frame,axis=1);
    dummy_df.set_index(dummy_df['code'],inplace=True);
    dummy_df.drop(['code','银行'],inplace=True,axis=1)
    # 如果所有哑变量全是0，代表行业是银行
    return dummy_df;
dummy_df = set_dummy();
# The dummy_df is a global var and will be used in 
# multi linear reg process 
#%%
def multi_reg_data(enddate):
    frees_series = frees_norm(enddate);
    pe_series, drop_list = pe_norm(enddate);
    temp_list_frees = [];
    temp_list_pe = [];
    for item in drop_list:
        if item not in dummy_df.index:
            drop_list.remove(item);
    for idx in dummy_df.index:
        if (idx not in pe_series.index) or (idx not in frees_series.index):
            temp_list_frees.append(float('nan'));
            temp_list_pe.append(float('nan'));
            drop_list.append(idx);
        else:
            temp_list_frees.append(frees_series[idx]);
            temp_list_pe.append(pe_series[idx]);
    temp_series_frees = pd.Series(temp_list_frees, name = 'Frees', index = dummy_df.index);
    temp_series_pe = pd.Series(temp_list_pe, name = 'PE', index = dummy_df.index);
    frame = [dummy_df, temp_series_frees, temp_series_pe];
    df = pd.concat(frame, axis = 1);
    df.drop(drop_list, axis = 0, inplace = True, errors='ignore');
    return df;
#%%
from sklearn.linear_model import LinearRegression
def multi_res_factor(enddate):
    enddate = enddate - (date_to_int('2005/1/4'));
    df = multi_reg_data(enddate);
    y = df.loc[:,'PE'].as_matrix(columns=None);
    y = np.array([y]).T;
    x = df.drop(['PE'],axis=1);
    x = x.as_matrix(columns=None);
    l = LinearRegression();
    l.fit(x,y);
    res = y-l.predict(x);
    final_list = [item[0] for item in res];
    final_series = pd.Series(final_list, index = df.index);
    final_series.sort_values(ascending=True, inplace=True);
    return final_series;
#%%
def frees_factor(enddate):
    frees_list = [];
    enddate = enddate - (date_to_int('2005/1/4'));
    for key,value in data_close.iteritems():
        if (key not in data_frees.columns):
            frees_list.append(float('nan'));
        elif (data_frees[key][enddate] <= 0.0):
            frees_list.append(float('nan'));
        else:
            frees_list.append(data_frees[key][enddate]);
    final_series = pd.Series(frees_list, index = stock_code);
    final_series.sort_values(ascending=True, inplace=True);
    return final_series; 
#%%    
def multi_frees_data(enddate):
    frees_series = frees_norm(enddate);
    temp_list_frees = [];
    drop_list = [];
    for idx in dummy_df.index:
        if (idx not in frees_series.index):
            temp_list_frees.append(float('nan'));
            drop_list.append(idx);
        else:
            temp_list_frees.append(frees_series[idx]);
    temp_series_frees = pd.Series(temp_list_frees, name = 'Frees', index = dummy_df.index);
    frame = [dummy_df, temp_series_frees];
    df = pd.concat(frame, axis = 1);
    df.drop(drop_list, axis = 0, inplace = True, errors='ignore');
    return df;
df = multi_frees_data(1000);
#%%
from sklearn.linear_model import LinearRegression
def multi_frees_factor(enddate):
    enddate = enddate - (date_to_int('2005/1/4'));
    df = multi_frees_data(enddate);
    y = df.loc[:,'Frees'].as_matrix(columns=None);
    y = np.array([y]).T;
    x = df.drop(['Frees'],axis=1);
    x = x.as_matrix(columns=None);
    l = LinearRegression();
    l.fit(x,y);
    res = y-l.predict(x);
    final_list = [item[0] for item in res];
    final_series = pd.Series(final_list, index = df.index);
    final_series.sort_values(ascending=True, inplace=True);
    return final_series;
#%%
def pb_factor(enddate):
    pb_list = [];
    enddate = enddate - (date_to_int('2005/1/4'));
    for key,value in data_close.iteritems():
        if (key not in data_pb.columns):
            pb_list.append(float('nan'));
        elif (data_pb[key][enddate] <= 0.0):
            pb_list.append(float('nan')); #delete negtive values
        else:
            pb_list.append(data_pb[key][enddate]);
    final_series = pd.Series(pb_list, index = stock_code);
    final_series.sort_values(ascending=True, inplace=True);
    return final_series; 
#%%
def pb_norm(enddate):
    drop_list = []; # stock with nagetive PB should be droped
    pb_list = [];
    for key,value in data_close.iteritems():
        if (key not in data_pb.columns):
            drop_list.append(key);
            pb_list.append(float('nan'));
        elif (data_pb[key][enddate] <= 0.0):
            drop_list.append(key);
            pb_list.append(float('nan'));
        else:
            pb_list.append(data_pb[key][enddate]);
    sigma = np.nanstd(pb_list);
    mu = np.nanmean(pb_list);
    norm_list = [];
    cap = mu + 1.5*sigma;
    for item in pb_list:
        if (item > cap):
            norm_list.append(1.5);
        else:
            norm_list.append((item - mu) / sigma);
    final_series = pd.Series(norm_list, index = stock_code);
    return final_series, drop_list;
#%%
def get_pb_reg_data(enddate):
    frees_series = frees_norm(enddate);
    pb_series, drop_list = pb_norm(enddate);
    frees_series.drop(drop_list, inplace = True, errors = 'ignore');
    pb_series.drop(drop_list, inplace = True);
    frame = [frees_series, pb_series];
    df = pd.concat(frame, axis=1);
    df.columns = ['x','y'];
    return df;    
#%%
from sklearn import linear_model
def pb_res_factor(enddate):
    enddate = enddate - (date_to_int('2005/1/4'));
    res_list = [];
    df = get_pb_reg_data(enddate);
    regr = linear_model.LinearRegression();
    regr.fit(df['x'].values.reshape(-1,1), df['y']);
    a, b = regr.coef_, regr.intercept_;
    stock_idx = df.index;
    for key in stock_idx:
        res = df['y'][key] - b*df['x'][key] - a;
        res_list.append(res[0]);
    final_series = pd.Series(res_list, index = stock_idx);
    final_series.sort_values(ascending=True, inplace=True);
    return final_series;     
#%%   
def pb_multi_reg_data(enddate):
    frees_series = frees_norm(enddate);
    pb_series, drop_list = pb_norm(enddate);
    temp_list_frees = [];
    temp_list_pb = [];
    for item in drop_list:
        if item not in dummy_df.index:
            drop_list.remove(item);
    for idx in dummy_df.index:
        if (idx not in pb_series.index) or (idx not in frees_series.index):
            temp_list_frees.append(float('nan'));
            temp_list_pb.append(float('nan'));
            drop_list.append(idx);
        else:
            temp_list_frees.append(frees_series[idx]);
            temp_list_pb.append(pb_series[idx]);
    temp_series_frees = pd.Series(temp_list_frees, name = 'Frees', index = dummy_df.index);
    temp_series_pb = pd.Series(temp_list_pb, name = 'PB', index = dummy_df.index);
    frame = [dummy_df, temp_series_frees, temp_series_pb];
    df = pd.concat(frame, axis = 1);
    df.drop(drop_list, axis = 0, inplace = True, errors='ignore');
    return df;    
#%%
from sklearn.linear_model import LinearRegression
def pb_multi_res_factor(enddate):
    enddate = enddate - (date_to_int('2005/1/4'));
    df = pb_multi_reg_data(enddate);
    y = df.loc[:,'PB'].as_matrix(columns=None);
    y = np.array([y]).T;
    x = df.drop(['PB'],axis=1);
    x = x.as_matrix(columns=None);
    l = LinearRegression();
    l.fit(x,y);
    res = y-l.predict(x);
    final_list = [item[0] for item in res];
    final_series = pd.Series(final_list, index = df.index);
    final_series.sort_values(ascending=True, inplace=True);
    return final_series;
#%%
def turnover_ascending(enddate):
    turn_list = [];
    enddate = enddate - (date_to_int('2005/1/4'));
    for key,value in data_close.iteritems():
        if (key not in data_turnover.columns):
            turn_list.append(float('nan'));
        elif (data_turnover[key][enddate] == 0.0):
            turn_list.append(float('nan')); #delete negtive values
        else:
            turn_list.append(data_turnover[key][enddate]);
    final_series = pd.Series(turn_list, index = stock_code);
    final_series.sort_values(ascending=True, inplace=True);
    return final_series;
#%%
def turnover_descending(enddate):
    turn_list = [];
    enddate = enddate - (date_to_int('2005/1/4'));
    for key,value in data_close.iteritems():
        if (key not in data_turnover.columns):
            turn_list.append(float('nan'));
        elif (data_turnover[key][enddate] == 0.0):
            turn_list.append(float('nan')); #delete negtive values
        else:
            turn_list.append(data_turnover[key][enddate]);
    final_series = pd.Series(turn_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
def amt_factor(enddate):
    amt_list = [];
    for key,value in data_amt.iteritems():
        if (data_amt[key][enddate] == 0.0):
            amt_list.append(float('nan')); #delete negtive values
        else:
            amt_list.append(data_amt[key][enddate]);
    final_series = pd.Series(amt_list, index = stock_code);
    final_series.sort_values(ascending=True, inplace=True);
    return final_series;
#%%
def volume_factor(enddate):
    volume_list = [];
    for key,value in data_volume.iteritems():
        if (data_volume[key][enddate] == 0.0):
            volume_list.append(float('nan')); #delete negtive values
        else:
            volume_list.append(data_volume[key][enddate]);
    final_series = pd.Series(volume_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%


stus_score = np.array([[80, 88], [82, 81], [84, 75], [86, 83], [75, 81]]) 
print("加分前:") 
print(stus_score) 
# 为所有平时成绩都加5分 
stus_score[:, 0] = stus_score[:, 0]+5 
print("加分后:") 
print(stus_score) 







