#%%
"""
@Project: alpha101

@FileName: Trials on alpha101

@Author：Yufei Gao

@Create date: 2018/7/6

@description：do some test on the sample alphas

@Update date：  

@Vindicator：  

"""  
#%%
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
#import multiprocessing
import matplotlib.pyplot as plt
import scipy.io
#import stats
from basic_func import Rank
from basic_func import Ts_argmax
from basic_func import scale 
#from basic_func import SignedPower
from basic_func  import *
#%%
#import the data table
mat = scipy.io.loadmat('C:/Users/HP/Desktop/data_gp.mat');
data_close=pd.DataFrame(mat['data']['Close'][0][0]);
data_close.head();
allastock = mat['AllAStockSet'];
data_all = pd.DataFrame(allastock);
data_all;
stock_code = [];
for i in range(3521):
    stock_code.append(data_all[i][1][0]);
data_close.columns=stock_code;
data_close.fillna(0, inplace=True);

data_volume = pd.DataFrame(mat['data']['Volume'][0][0]);
data_volume.columns=stock_code;
data_volume.fillna(0, inplace=True);
data_volume.head();

data_open = pd.DataFrame(mat['data']['Open'][0][0]);
data_open.columns=stock_code;
data_open.fillna(0, inplace=True);
data_open.head();

data_low = pd.DataFrame(mat['data']['Low'][0][0]);
data_low.columns=stock_code;
data_low.fillna(0, inplace=True);
data_low.head();

data_high = pd.DataFrame(mat['data']['High'][0][0]);
data_high.columns=stock_code;
data_high.fillna(0, inplace=True);
data_high.head();

data_amt = pd.DataFrame(mat['data']['Amt'][0][0]);
data_amt.columns=stock_code;
data_amt.fillna(0, inplace=True);
data_amt.head();

frame = [];
data_vwap = data_volume.copy();
for key,value in data_vwap.iteritems():
    frame.append(data_amt[key]/data_volume[key]);
data_vwap = pd.concat(frame, axis=1);
data_vwap.fillna(0, inplace=True);
data_vwap.head();    
#%%
#Alpha101
'''
alpha_001(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
'''
def alpha001(enddate):
    Ticker_list = [];
    # cpt the return table in the last 25 days
    ret = data_close[:][enddate-25:enddate].copy();
    for key,value in ret.iteritems():
        for i in range(enddate-25, enddate):
            if (data_close[key][i-1] == 0)|(data_close[key][i] == 0):
                ret[key][i] = float('nan');
            else:
                ret[key][i] = np.log(data_close[key][i]) - np.log(data_close[key][i-1]);
    
    #cpt the value based on the ret table
    rank_list = [];
    #loop over the universe
    for key,value in ret.iteritems():
        Ticker_list.append(key);
        Ts_arglist = [];
        # cpt the Ts_
        for day in range(5,0,-1):            
            if (ret[key][enddate-day] < 0):
                temp_val = np.std(ret[key][enddate-day-19:enddate-day+1]);
            else:
                temp_val = data_close[key][enddate-day];
            Ts_arglist.append(SignedPower([temp_val],2.0)[0]);
        rank_list.append(Ts_argmax(Ts_arglist,5));        
    rank_val = Rank(rank_list);
    final_val = [rank_val[i]-0.5 for i in range(len(rank_val))];
    final_series = pd.Series(final_val, index = Ticker_list);
    final_series.sort_values(ascending = False, inplace = True);
    return final_series;
#%%
sig = alpha001(40)
sig.head(20)
#%%
'''
alpha_002(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1*correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
'''
'''
need to use 6 lists to cpt last 6 day's rank for each stock
Then creat lists to contain past 6 days' rank value, cpt corr

'''
#%%
def alpha002(enddate):
    #cpt 1st item
    rank_list = [[] for i in range(6)];
    rank_Series = [[] for i in range(6)];
    for day in range(6,0,-1):
        for key,value in data_volume.iteritems():
            if (data_volume[key][enddate-day] == 0)|(data_volume[key][enddate-day-2] == 0):
                delta_vol = float('nan');
            else:
                delta_vol = np.log(data_volume[key][enddate-day])-np.log(data_volume[key][enddate-day-2]);
            rank_list[day-1].append(delta_vol);
        rank_list[day-1] = Rank(rank_list[day-1]);
        rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
    frame1 = [rank_Series[i] for i in range(6)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    rank_list = [[] for i in range(6)];
    rank_Series = [[] for i in range(6)];
    for day in range(6,0,-1):
        for key,value in data_volume.iteritems():
            if (data_open[key][enddate-day] == 0)|(data_close[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = (data_close[key][enddate-day]-data_open[key][enddate-day])/(data_open[key][enddate-day]);
            rank_list[day-1].append(val);
        rank_list[day-1] = Rank(rank_list[day-1]);
        rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
    frame2 = [rank_Series[i] for i in range(6)];
    second_frame = pd.concat(frame2, axis=1);
    
    #cpt the corr
    corr_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        corr_list.append(-1*s1.corr(s2));
    final_series = pd.Series(corr_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
sig=alpha002(100)
#%%
'''
alpha_003(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1*correlation(rank(open), rank(volume), 10))
'''
#%%
def alpha003(enddate):
    #cpt 1st item
    rank_list = [[] for i in range(10)];
    rank_Series = [[] for i in range(10)];
    for day in range(10,0,-1):
        for key,value in data_volume.iteritems():
            if (data_volume[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_volume[key][enddate-day];
            rank_list[day-1].append(val);
        rank_list[day-1] = Rank(rank_list[day-1]);
        rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
    frame1 = [rank_Series[i] for i in range(10)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    rank_list = [[] for i in range(10)];
    rank_Series = [[] for i in range(10)];
    for day in range(10,0,-1):
        for key,value in data_volume.iteritems():
            if (data_open[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_open[key][enddate-day];
            rank_list[day-1].append(val);
        rank_list[day-1] = Rank(rank_list[day-1]);
        rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
    frame2 = [rank_Series[i] for i in range(10)];
    second_frame = pd.concat(frame2, axis=1);
    
    #cpt the corr
    corr_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        corr_list.append(-1*s1.corr(s2));
    final_series = pd.Series(corr_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%
'''
alpha_004(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1*Ts_Rank(rank(low), 9))
'''
#%%
def alpha004(enddate):
    # cpt the rank of low in past 9 days
    rank_list = [[] for i in range(9)];
    rank_Series = [[] for i in range(10)];
    ts_rank_list = [];
    for day in range(9,0,-1):
        for key,value in data_volume.iteritems():
            if (data_low[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_low[key][enddate-day];
            rank_list[day-1].append(val);
        rank_list[day-1] = Rank(rank_list[day-1]);
        rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
    frame = [rank_Series[i] for i in range(9)];
    df = pd.concat(frame, axis=1);
    for i in range(len(df)):
        rank_list = list(df.iloc[i][:]);
        ts_rank_list.append(-1*Rank(rank_list)[0]);
    ts_Series = pd.Series(ts_rank_list, index = stock_code);
    ts_Series.sort_values(ascending=False, inplace=True);
    return ts_Series;
#%%
'''
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(rank((open - (sum(vwap, 10) / 10)))*(-1*abs(rank((close - vwap)))))
'''
#%%
def alpha005(enddate):
    temp1 = [];
    temp2 = [];
    for key,value in data_vwap.iteritems():
        val = sum([data_vwap[key][enddate-i] for i in range(10)])/10.0;
        temp1.append(data_open[key][enddate] - val);
        temp2.append(data_close[key][enddate]-data_vwap[key][enddate]);
    rk1 = Rank(temp1);
    rk2 = Rank(temp2);
    final_list = [rk1[i]*(-abs(rk2[i])) for i in range(len(stock_code))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_006(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1*correlation(open, volume, 10))
'''
#%%
def alpha006(enddate):
    #cpt 1st item
    val_list = [[] for i in range(10)];
    val_Series = [[] for i in range(10)];
    for day in range(10,0,-1):
        for key,value in data_volume.iteritems():
            if (data_volume[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_volume[key][enddate-day];
            val_list[day-1].append(val);
        val_Series[day-1] = pd.Series(val_list[day-1], index = stock_code);
    frame1 = [val_Series[i] for i in range(10)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    val_list = [[] for i in range(10)];
    val_Series = [[] for i in range(10)];
    for day in range(10,0,-1):
        for key,value in data_volume.iteritems():
            if (data_open[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_open[key][enddate-day];
            val_list[day-1].append(val);
        val_Series[day-1] = pd.Series(val_list[day-1], index = stock_code);
    frame2 = [val_Series[i] for i in range(10)];
    second_frame = pd.concat(frame2, axis=1);
    
    #cpt the corr
    corr_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        corr_list.append(-1*s1.corr(s2));
    final_series = pd.Series(corr_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 

#%%
'''
alpha_008(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1*rank(((sum(open, 5)*sum(returns, 5)) - delay((sum(open, 5)*sum(returns, 5)), 10))))
'''
#%%
def alpha008(enddate):
    rank_list = [];
    final_rank = [];
    # cpt the return table
    ret = data_close[:][enddate-15:enddate].copy();
    for key,value in ret.iteritems():
        for i in range(enddate-15, enddate):
            if (data_close[key][i-1] == 0)|(data_close[key][i] == 0):
                ret[key][i] = float('nan');
            else:
                ret[key][i] = np.log(data_close[key][i]) - np.log(data_close[key][i-1]);
    for key,value in ret.iteritems():
        val_cur = sum([data_open[key][enddate-i] for i in range(5,0,-1)])*sum([ret[key][enddate-i] for i in range(5,0,-1)]);
        val_delay = sum([data_open[key][enddate-10-i] for i in range(5,0,-1)])*sum([ret[key][enddate-10-i] for i in range(5,0,-1)]);
        rank_list.append(val_cur-val_delay);
    final_rank = Rank(rank_list);
    final_Series = pd.Series(final_rank, index = stock_code);
    final_Series.sort_values(ascending=False, inplace=True);
    return final_Series;
#%%
'''
alpha_009(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1*delta(close, 1))))
'''   
#%%
def alpha009(enddate):
    final_list = [];
    for key,value in data_close.iteritems():
        lag_close = data_close[key][enddate] - data_close[key][enddate-1];
        lag_list = [data_close[key][enddate-i] - data_close[key][enddate-i-1] for i in range(5)];
        lag_min = min(lag_list);
        lag_max = max(lag_list);
        if (lag_min > 0):
            val = lag_close;
        elif (lag_max > 0):
            val = lag_close;
        else:
            val = -1*lag_close;
        final_list.append(val);
    final_Series = pd.Series(final_list, index = stock_code);
    final_Series.sort_values(ascending=False, inplace=True);
    return final_Series;
#%%
'''
alpha_010(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1*delta(close, 1)))))
'''    
#%%
def alpha010(enddate):   
    rank_list = [];
    for key,value in data_close.iteritems():
        lag_close = data_close[key][enddate] - data_close[key][enddate-1];
        lag_list = [data_close[key][enddate-i] - data_close[key][enddate-i-1] for i in range(4)];
        lag_min = min(lag_list);
        lag_max = max(lag_list);
        if (lag_min > 0):
            val = lag_close;
        elif (lag_max > 0):
            val = lag_close;
        else:
            val = -1*lag_close;
        rank_list.append(val);
    final_Series = pd.Series(Rank(rank_list), index = stock_code);
    final_Series.sort_values(ascending=False, inplace=True);
    return final_Series;
#%%
'''alpha_011(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3)))*rank(delta(volume, 3))
'''
#%%
def alpha011(enddate):
    temp1 = [];
    temp2 = [];
    temp3 = [];
    for key,value in data_vwap.iteritems():
        temp1.append(max(data_vwap[key][enddate-i]-data_close[key][enddate-i] for i in range(3)));
        temp2.append(min(data_vwap[key][enddate-i]-data_close[key][enddate-i] for i in range(3)));
        temp3.append(data_volume[key][enddate]-data_volume[key][enddate-3]);
    rk1 = Rank(temp1);
    rk2 = Rank(temp2);
    rk3 = Rank(temp3);
    final_list = [(rk1[i]+rk2[i])*rk3[i] for i in range(len(stock_code))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(sign(delta(volume, 1))*(-1 * delta(close, 1)))
'''
#%%
def alpha012(enddate):
    final_list = [];
    for key,value in data_close.iteritems():
        lag_close = data_close[key][enddate] - data_close[key][enddate-1];
        lag_vol = data_volume[key][enddate] - data_volume[key][enddate-1];
        val = (-lag_close)*(np.sign(lag_vol));
        final_list.append(val);
    final_Series = pd.Series(final_list, index = stock_code);
    final_Series.sort_values(ascending=False, inplace=True);
    return final_Series;   
#%%
'''
alpha_013(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1*rank(covariance(rank(close), rank(volume), 5)))
'''
#%%
def alpha013(enddate):
    #cpt 1st item
    rank_list = [[] for i in range(5)];
    rank_Series = [[] for i in range(5)];
    for day in range(5,0,-1):
        for key,value in data_volume.iteritems():
            if (data_volume[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_volume[key][enddate-day];
            rank_list[day-1].append(val);
        rank_list[day-1] = Rank(rank_list[day-1]);
        rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
    frame1 = [rank_Series[i] for i in range(5)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    rank_list = [[] for i in range(5)];
    rank_Series = [[] for i in range(5)];
    for day in range(5,0,-1):
        for key,value in data_volume.iteritems():
            if (data_close[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_close[key][enddate-day];
            rank_list[day-1].append(val);
        rank_list[day-1] = Rank(rank_list[day-1]);
        rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
    frame2 = [rank_Series[i] for i in range(5)];
    second_frame = pd.concat(frame2, axis=1);
    
    #cpt the cov
    cov_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        cov_list.append(s1.cov(s2));
    cov_list = Rank(cov_list);
    final_list = [-cov_list[i] for i in range(len(cov_list))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%
'''
alpha_014(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
((-1*rank(delta(returns, 3)))*correlation(open, volume, 10))
'''    
#%%    
def alpha014(enddate):
    #cpt 1st item
    first_list = [[] for i in range(10)];
    first_Series = [[] for i in range(10)];
    for day in range(10,0,-1):
        for key,value in data_volume.iteritems():
            if (data_volume[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_volume[key][enddate-day];
            first_list[day-1].append(val);
        first_Series[day-1] = pd.Series(first_list[day-1], index = stock_code);
    frame1 = [first_Series[i] for i in range(10)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    second_list = [[] for i in range(10)];
    second_Series = [[] for i in range(10)];
    for day in range(10,0,-1):
        for key,value in data_volume.iteritems():
            if (data_open[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_open[key][enddate-day];
            second_list[day-1].append(val);
        second_Series[day-1] = pd.Series(second_list[day-1], index = stock_code);
    frame2 = [second_Series[i] for i in range(10)];
    second_frame = pd.concat(frame2, axis=1);
    
    #cpt the corr
    corr_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        corr_list.append(-1*s1.corr(s2));
        
    #cpt the rank term
    delta_list = [];
    for key,value in data_close.iteritems():
        if (data_close[key][enddate] == 0) | (data_close[key][enddate-1] == 0) | (data_close[key][enddate-3] == 0) | (data_close[key][enddate-4] == 0):
            delta_list.append(float('nan'));
        else:                                                                                                                                   
            delta_list.append((np.log(data_close[key][enddate])-np.log(data_close[key][enddate-1]))-(np.log(data_close[key][enddate-3])-np.log(data_close[key][enddate-4])));
    rank_list = Rank(delta_list);
    rank_list = [-rank_list[i] for i in range(len(rank_list))];
    final_list = [a*b for a,b in zip(rank_list,corr_list)];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%
'''
alpha_015(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1*sum(rank(correlation(rank(high), rank(volume), 3)), 3))
'''
#%%
def alpha015(enddate):
    rk = [[],[],[]];
    for loop in range(3):        
        #cpt 1st item
        rank_list = [[] for i in range(5)];
        rank_Series = [[] for i in range(5)];
        for day in range(5,0,-1):
            for key,value in data_volume.iteritems():
                if (data_volume[key][enddate-day] == 0):
                    val = float('nan');
                else:
                    val = data_volume[key][enddate-day];
                rank_list[day-1].append(val);
            rank_list[day-1] = Rank(rank_list[day-1]);
            rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
        frame1 = [rank_Series[i] for i in range(5)];
        first_frame = pd.concat(frame1, axis=1);
    
        #cpt 2nd item
        rank_list = [[] for i in range(5)];
        rank_Series = [[] for i in range(5)];
        for day in range(5,0,-1):
            for key,value in data_volume.iteritems():
                if (data_high[key][enddate-day] == 0):
                    val = float('nan');
                else:
                    val = data_high[key][enddate-day];
                rank_list[day-1].append(val);
            rank_list[day-1] = Rank(rank_list[day-1]);
            rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
        frame2 = [rank_Series[i] for i in range(5)];
        second_frame = pd.concat(frame2, axis=1);
    
        #cpt the corr
        corr_list = [];
        for i in range(len(stock_code)):
            s1 = first_frame.iloc[i][:];
            s2 = second_frame.iloc[i][:];
            corr_list.append(s1.corr(s2));
        corr_list = Rank(corr_list);
        rk[loop] = Rank(corr_list);
        enddate = enddate - 1;
        
    final_list = [-(rk[0][i]+rk[1][i]+rk[2][i]) for i in range(len(stock_code))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%
'''
alpha_016(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1*rank(covariance(rank(high), rank(volume), 5)))
'''
#%%
def alpha016(enddate):
    #cpt 1st item
    rank_list = [[] for i in range(5)];
    rank_Series = [[] for i in range(5)];
    for day in range(5,0,-1):
        for key,value in data_volume.iteritems():
            if (data_volume[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_volume[key][enddate-day];
            rank_list[day-1].append(val);
        rank_list[day-1] = Rank(rank_list[day-1]);
        rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
    frame1 = [rank_Series[i] for i in range(5)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    rank_list = [[] for i in range(5)];
    rank_Series = [[] for i in range(5)];
    for day in range(5,0,-1):
        for key,value in data_volume.iteritems():
            if (data_high[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_high[key][enddate-day];
            rank_list[day-1].append(val);
        rank_list[day-1] = Rank(rank_list[day-1]);
        rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
    frame2 = [rank_Series[i] for i in range(5)];
    second_frame = pd.concat(frame2, axis=1);
    
    #cpt the cov
    cov_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        cov_list.append(s1.cov(s2));
    cov_list = Rank(cov_list);
    final_list = [-cov_list[i] for i in range(len(cov_list))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%
#%%
'''
alpha_018(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1*rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
'''
#%%
def alpha018(enddate):
    #cpt the corr item
    #cpt 1st item
    first_list = [[] for i in range(10)];
    first_Series = [[] for i in range(10)];
    for day in range(10,0,-1):
        for key,value in data_close.iteritems():
            if (data_close[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_close[key][enddate-day];
            first_list[day-1].append(val);
        first_Series[day-1] = pd.Series(first_list[day-1], index = stock_code);
    frame1 = [first_Series[i] for i in range(10)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    second_list = [[] for i in range(10)];
    second_Series = [[] for i in range(10)];
    for day in range(10,0,-1):
        for key,value in data_close.iteritems():
            if (data_open[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_open[key][enddate-day];
            second_list[day-1].append(val);
        second_Series[day-1] = pd.Series(first_list[day-1], index = stock_code);
    frame2 = [second_Series[i] for i in range(10)];
    second_frame = pd.concat(frame2, axis=1);
    
    #cpt the corr
    corr_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        corr_list.append(s1.corr(s2)); 
        
    #cpt the std term
    stddev = [];
    diff = [];
    for key,value in data_close.iteritems():
        temp = [];
        for i in range(5):
            val = data_close[key][enddate-i]-data_open[key][enddate-i];
            temp.append(abs(val));
        stddev.append(np.std(temp));
        diff.append(val);
    
    #cpt rank
    rank_list = Rank([stddev[i]+diff[i]+corr_list[i] for i in range(len(stock_code))]);
    
    final_list = [-rank_list[i] for i in range(len(stock_code))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_019(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
((-1*sign(((close - delay(close, 7)) + delta(close, 7))))*(1 + rank((1 + sum(returns, 250)))))
'''
#%%
def alpha019(enddate):
    # cpt the return table in the last 250 days
    rank_list = [];
    ret = data_close[:][enddate-250:enddate].copy();
    for key,value in ret.iteritems():
        for i in range(enddate-250, enddate):
            if (data_close[key][i-1] == 0)|(data_close[key][i] == 0):
                ret[key][i] = float('nan');
            else:
                ret[key][i] = np.log(data_close[key][i]) - np.log(data_close[key][i-1]);
                
    # cpt the rank term
    temp = [];
    for key,value in ret.iteritems():
        temp.append(1 + sum(ret[key][:]));
    temp = Rank(temp);
    rank_list=[i+1 for i in temp];
        
    #cpt the 1st term
    first_list = [];
    for key,value in data_close.iteritems():
        delta = data_close[key][enddate] - data_close[key][enddate-7];
        first_list.append(-1*np.sign(data_close[key][enddate]-data_close[key][enddate-7]+delta));
        
    final_list = [a*b for a,b in zip(rank_list,first_list)];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%
'''
alpha_020(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(((-1*rank((open - delay(high, 1))))* rank((open - delay(close, 1))))* rank((open - delay(low, 1))))
'''
#%%
def alpha020(enddate):
    temp1 = [];
    temp2 = [];
    temp3 = [];
    for key,value in data_close.iteritems():
        temp1.append(data_open[key][enddate] - data_high[key][enddate-1]);
        temp2.append(data_open[key][enddate] - data_close[key][enddate-1]);
        temp3.append(data_open[key][enddate] - data_low[key][enddate-1]);
    rk1 = Rank(temp1);
    rk2 = Rank(temp2);
    rk3 = Rank(temp3);
    
    final_list = [-rk1[i]*rk2[i]*rk3[i] for i in range(len(rk1))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
#%%
'''
alpha_022(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1* (delta(correlation(high, volume, 5), 5)* rank(stddev(close, 20))))
'''
#%%
def alpha022(enddate):
    # cpt the delta term
    delta_item = [[],[]];
    for loop in range(2):      
        #cpt the corr item
        #cpt 1st item
        first_list = [[] for i in range(5)];
        first_Series = [[] for i in range(5)];
        for day in range(5,0,-1):
            for key,value in data_high.iteritems():
                if (data_high[key][enddate-day] == 0):
                    val = float('nan');
                else:
                    val = data_high[key][enddate-day];
                first_list[day-1].append(val);
            first_Series[day-1] = pd.Series(first_list[day-1], index = stock_code);
        frame1 = [first_Series[i] for i in range(5)];
        first_frame = pd.concat(frame1, axis=1);
    
        #cpt 2nd item
        second_list = [[] for i in range(5)];
        second_Series = [[] for i in range(5)];
        for day in range(5,0,-1):
            for key,value in data_high.iteritems():
                if (data_volume[key][enddate-day] == 0):
                    val = float('nan');
                else:
                    val = data_volume[key][enddate-day];
                second_list[day-1].append(val);
            second_Series[day-1] = pd.Series(first_list[day-1], index = stock_code);
        frame2 = [second_Series[i] for i in range(5)];
        second_frame = pd.concat(frame2, axis=1);
    
        #cpt the corr
        for i in range(len(stock_code)):
            s1 = first_frame.iloc[i][:];
            s2 = second_frame.iloc[i][:];
            delta_item[loop].append(s1.corr(s2));
        
        enddate = enddate - 5;
    
    delta_list = [delta_item[0][i] - delta_item[1][i] for i in range(len(delta_item[0]))];
    
    # cpt the rank term
    stddev = [];
    for key,value in data_close.iteritems():
        temp = [];
        for i in range(20):
            val = data_close[key][enddate-i];
            temp.append(val);
        stddev.append(np.nanstd(temp));
    rank_list = Rank(stddev);
    
    final_list = [-delta_list[i]*rank_list[i] for i in range(len(delta_list))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_024(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) <= 0.05) ? (-1* (close - ts_min(close, 100))) : (-1* delta(close, 3)))
'''
#%%
#%%
'''
alpha_023(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个Series：index为成分股代码，values为对应因子值或0.00，当不满足条件时为0.00
因子公式： 
(((sum(high, 20) / 20) < high) ? (-1* delta(high, 2)) : 0)
'''
#%%
def alpha023(enddate):
    final_list = [];
    for key,value in data_high.iteritems():
        val = sum([data_high[key][enddate-i] for i in range(20)])/20.0;
        if (val < data_high[key][enddate]):
            delta = data_high[key][enddate] - data_high[key][enddate - 2];
            final_list.append(-delta);
        else:
            final_list.append(0);
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_024(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) <= 0.05) ? (-1* (close - ts_min(close, 100))) : (-1* delta(close, 3)))
'''
#%%
def alpha024(enddate):
    final_list = [];
    for key,value in data_close.iteritems():
        cur_val = sum([data_close[key][enddate-i] for i in range(100)])/100.0;
        past_val = sum([data_close[key][enddate-i] for i in range(100)])/100.0;
        delta = cur_val - past_val;
        if (delta/data_close[key][enddate-100] <= 0.5):
            min_val = min([data_close[key][enddate-i] for i in range(100)]);
            final_list.append(-(data_close[key][enddate] - min_val));
        else:
            final_list.append(-(data_close[key][enddate] - data_close[key][enddate-3]));
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
#%%
#%%
'''
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1* ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
'''
#%%
def alpha026_ts_rank(enddate):
    rank_list = [[] for i in range(5)];
    for day in range(5,0,-1):
        for key,value in data_volume.iteritems():
            if (data_volume[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_volume[key][enddate-day];
            rank_list[day-1].append(val);
    ts_rank_1 = [];
    for i in range(len(stock_code)):
        temp = [rank_list[day-1][i] for day in range(5,0,-1)];
        ts_rank_1.append(Rank(temp)[0]);
    
    #cpt 2nd item
    rank_list = [[] for i in range(5)];
    for day in range(5,0,-1):
        for key,value in data_volume.iteritems():
            if (data_high[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_high[key][enddate-day];
            rank_list[day-1].append(val);
    ts_rank_2 = [];
    for i in range(len(stock_code)):
        temp = [rank_list[day-1][i] for day in range(5,0,-1)];
        ts_rank_2.append(Rank(temp)[0]);    
    return ts_rank_1, ts_rank_2;

def alpha026_corr(enddate):
    list_1 = [[] for i in range(5)];
    list_2 = [[] for i in range(5)];
    for day in range(5,0,-1):
        list_1[day-1],list_2[day-1] = alpha026_ts_rank(enddate-day);
    ts_max_list = [];
    for i in range(len(stock_code)):
        temp1 = pd.Series([list_1[day][i] for day in range(5)]);
        temp2 = pd.Series([list_2[day][i] for day in range(5)]);
        ts_max_list.append(temp1.corr(temp2));
    return ts_max_list;

def alpha026(enddate):
    final_list = [];
    ts_max = [[] for i in range(3)];
    for day in range(3,0,-1):
        ts_max[day-1] = alpha026_corr(enddate-day);
    for i in range(len(stock_code)):
        final_list.append(-max([ts_max[day][i] for day in range(3)]));
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_027(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个Series：index为成分股代码，values为1或-1，满足条件为-1，不满足为1
因子公式： 
((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1* 1) : 1)
'''
#%%
def alpha027(enddate):
    cr = [[],[]];
    for loop in range(2):
        #cpt 1st item
        rank_list = [[] for i in range(6)];
        rank_Series = [[] for i in range(6)];
        for day in range(6,0,-1):
            for key,value in data_volume.iteritems():
                if (data_volume[key][enddate-day] == 0):
                    val = float('nan');
                else:
                    val = data_volume[key][enddate-day];
                rank_list[day-1].append(val);
            rank_list[day-1] = Rank(rank_list[day-1]);
            rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
        frame1 = [rank_Series[i] for i in range(6)];
        first_frame = pd.concat(frame1, axis=1);
    
        #cpt 2nd item
        rank_list = [[] for i in range(6)];
        rank_Series = [[] for i in range(6)];
        for day in range(6,0,-1):
            for key,value in data_volume.iteritems():
                if (data_vwap[key][enddate-day] == 0):
                    val = float('nan');
                else:
                    val = data_vwap[key][enddate-day];
                rank_list[day-1].append(val);
            rank_list[day-1] = Rank(rank_list[day-1]);
            rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
        frame2 = [rank_Series[i] for i in range(6)];
        second_frame = pd.concat(frame2, axis=1);    
    
        #cpt the corr
        corr_list = [];
        for i in range(len(stock_code)):
            s1 = first_frame.iloc[i][:];
            s2 = second_frame.iloc[i][:];
            corr_list.append(s1.corr(s2));
        cr[loop] = corr_list;
            
        enddate = enddate - 1;
    
    # cpt the rank
    rank_list = Rank([(cr[0][i]+cr[1][i])/2.0 for i in range(len(cr[0]))]);    
    final_list = [];
    for i in range(len(rank_list)):
        if (rank_list[i] > 0.5):
            final_list.append(-1);
        else:
            final_list.append(1);
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_030(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3))))))* sum(volume, 5)) / sum(volume, 20))
'''
#%%
def alpha030(enddate):
    sign1 = [];
    sign2 = [];
    sign3 = [];
    sum1 = [];
    sum2 = [];
    for key,value in data_close.iteritems():
        sign1.append(np.sign(data_close[key][enddate]-data_close[key][enddate-1]));
        sign2.append(np.sign(data_close[key][enddate-1]-data_close[key][enddate-2]));
        sign3.append(np.sign(data_close[key][enddate-2]-data_close[key][enddate-3]));
        sum1.append(sum([data_volume[key][enddate-i] for i in range(5)]));
        sum2.append(sum([data_volume[key][enddate-i] for i in range(20)]));
    rank_list = Rank([sign1[i]+sign2[i]+sign3[i] for i in range(len(sign1))]);
    final_list = [(1.0-rank_list[i])*sum1[i]/float(sum2[i]) for i in range(len(stock_code))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%
'''
alpha_032(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(scale(((sum(close, 7) / 7) - close)) + (20* scale(correlation(vwap, delay(close, 5), 230))))
'''
#%%
def alpha032(enddate):
    # cpt the 1st term
    temp = [];
    for key,value in data_close.iteritems():
        temp.append(sum([data_close[key][enddate-i] for i in range(7)])/7.0 - data_close[key][enddate]);
    scl = scale(temp);
    
    # cpt the corr term
    #cpt 1st item
    first_list = [[] for i in range(230)];
    first_Series = [[] for i in range(230)];
    for day in range(230,0,-1):
        for key,value in data_vwap.iteritems():
            if (data_high[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_vwap[key][enddate-day];
            first_list[day-1].append(val);
        first_Series[day-1] = pd.Series(first_list[day-1], index = stock_code);
    frame1 = [first_Series[i] for i in range(230)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    second_list = [[] for i in range(230)];
    second_Series = [[] for i in range(230)];
    for day in range(230,0,-1):
        for key,value in data_vwap.iteritems():
            if (data_close[key][enddate-5-day] == 0):
                val = float('nan');
            else:
                val = data_close[key][enddate-5-day];
            second_list[day-1].append(val);
        second_Series[day-1] = pd.Series(first_list[day-1], index = stock_code);
    frame2 = [second_Series[i] for i in range(230)];
    second_frame = pd.concat(frame2, axis=1);
    
    #cpt the corr
    corr_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        corr_list.append(s1.corr(s2));
    corr_list = scale(corr_list);
    
    final_list = [(scl[i])+20*corr_list[i] for i in range(len(stock_code))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%
'''
alpha_033(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
rank((-1* ((1 - (open / close))^1)))
'''
#%%
def alpha033(enddate):
    rank_list = [];
    for key,value in data_close.iteritems():
        rank_list.append(-1 * (1 - (data_open[key][enddate] / data_close[key][enddate])));
    final_list = Rank(rank_list);
    final_series = pd.Series(final_list, index = stock_code); 
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_034(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
'''
#%%
def alpha034(enddate):
    #cpt the 1st term
    rank_std_list = [];
    dlt_list = [];
    for key,value in data_close.iteritems():
        ret = [];
        for i in range(5):
            if ((data_close[key][enddate-i] == 0) | (data_close[key][enddate-i-1] == 0)):
                ret.append(float('nan'));
            else:
                ret.append(np.log(data_close[key][enddate-i]) - np.log(data_close[key][enddate-i-1]));
        temp1 = [ret[0],ret[1]];
        temp2 = [ret[j] for j in range(5)];
        std1 = np.std(temp1);
        std2 = np.std(temp2);
        dlt_list.append(data_close[key][enddate] - data_close[key][enddate-1]);
        if (std2 == 0):
            rank_std_list.append(float('nan'));
        else:
            rank_std_list.append(std1/std2);
    rank_list = Rank(rank_std_list);
    rank_first_list = [1 - item for item in rank_list];
    first_item  = Rank(rank_first_list);
    
    #cpt the 2nd term
    rank_second_list = Rank(dlt_list);
    second_item = [1 - item for item in rank_second_list];
    
    final_list = [first_item[item] + second_item[item] for item in range(len(first_item))];
    final_series = pd.Series(final_list, index = stock_code); 
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_035(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
((Ts_Rank(volume, 32)* (1 - Ts_Rank(((close + high) - low), 16)))* (1 - Ts_Rank(returns, 32)))
'''
#%%
def alpha035(enddate):
    final_list = [];
    for key,value in data_close.iteritems():
        temp1 = [data_volume[key][enddate - i] for i in range(32)];
        temp2 = [data_close[key][enddate - i]+data_high[key][enddate - i]-data_low[key][enddate - i] for i in range(16)];
        ret = [];
        for i in range(32):
            if ((data_close[key][enddate-i] == 0) | (data_close[key][enddate-i-1] == 0)):
                ret.append(float('nan'));
            else:
                ret.append(np.log(data_close[key][enddate-i]) - np.log(data_close[key][enddate-i-1]));
        rank_1st_term = (Rank(temp1))[0];
        rank_2nd_term = (Rank(temp2))[0];
        rank_3rd_term = (Rank(ret))[0];
        final_list.append(rank_1st_term*(1-rank_2nd_term)*(1-rank_3rd_term));
    
    final_series = pd.Series(final_list, index = stock_code); 
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_037(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
'''
#%%
def alpha037(enddate):
    #cpt 1st item
    dly_list = [[] for i in range(200)];
    dly_Series = [[] for i in range(200)];
    for day in range(200,0,-1):
        for key,value in data_close.iteritems():
            val = (data_open[key][enddate-day-1]-data_close[key][enddate-day-1]);
            dly_list[day-1].append(val);
        dly_Series[day-1] = pd.Series(dly_list[day-1], index = stock_code);
    frame1 = [dly_Series[i] for i in range(200)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    close_list = [[] for i in range(200)];
    close_Series = [[] for i in range(200)];
    for day in range(200,0,-1):
        for key,value in data_close.iteritems():
            val = data_close[key][enddate-day];
            close_list[day-1].append(val);
        close_Series[day-1] = pd.Series(close_list[day-1], index = stock_code);
    frame1 = [close_Series[i] for i in range(200)];
    second_frame = pd.concat(frame1, axis=1);
    
    #cpt the corr
    corr_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        corr_list.append(s1.corr(s2));
    
    #cpt the rank term
    rank_list = [];
    for key,value in data_close.iteritems():
        rank_list.append(data_open[key][enddate] - data_open[key][enddate]);
    
    second_list = Rank(rank_list);        
    first_list = Rank(corr_list);
    final_list = [first_list[i]+second_list[i] for i in range(len(first_list))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%
'''
alpha_038(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
((-1* rank(Ts_Rank(close, 10)))* rank((close / open)))
'''
#%%
def alpha038(enddate):
    rank_1st_list = [];
    rank_2nd_list = [];
    for key,value in data_close.iteritems():
        temp = [data_close[key][enddate - i] for i in range(10)];
        rank_1st_list.append((Rank(temp))[0]);
        if ((data_close[key][enddate] == 0) | (data_open[key][enddate] == 0)):
            rank_2nd_list.append(float('nan'));
        else:
            rank_2nd_list.append(data_close[key][enddate]/data_open[key][enddate]);
    
    first_item = Rank(rank_1st_list);
    second_item = Rank(rank_2nd_list);
    
    final_list = [-1*first_item[i]*second_item[i] for i in range(len(first_item))];
    final_series = pd.Series(final_list, index = stock_code); 
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_040

alpha_040(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
((-1* rank(stddev(high, 10)))* correlation(high, volume, 10))
'''
#%%
def alpha040(enddate):
    #cpt the rank term
    rank_list = [];
    for key,value in data_high.iteritems():
        temp = [data_high[key][enddate-i] for i in range(10)];
        rank_list.append(np.std(temp));
    first_list = Rank(rank_list);
        
    #cpt 1st item
    high_list = [[] for i in range(10)];
    high_Series = [[] for i in range(10)];
    for day in range(10,0,-1):
        for key,value in data_close.iteritems():
            high_list[day-1].append(data_high[key][enddate]);
        high_Series[day-1] = pd.Series(high_list[day-1], index = stock_code);
    frame1 = [high_Series[i] for i in range(10)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    volume_list = [[] for i in range(10)];
    volume_Series = [[] for i in range(10)];
    for day in range(10,0,-1):
        for key,value in data_close.iteritems():
            volume_list[day-1].append(data_volume[key][enddate]);
        volume_Series[day-1] = pd.Series(volume_list[day-1], index = stock_code);
    frame1 = [volume_Series[i] for i in range(10)];
    second_frame = pd.concat(frame1, axis=1);
    
    #cpt the corr
    corr_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        corr_list.append(s1.corr(s2));
        
    final_list = [-first_list[i]*corr_list[i] for i in range(len(first_list))];
    final_series = pd.Series(final_list, index = stock_code); 
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_041(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(((high* low)^0.5) - vwap)
'''
#%%
def alpha041(enddate):
    final_list = [];
    for key,value in data_close.iteritems():    
        final_list.append((data_high[key][enddate]*data_low[key][enddate])**0.5 - data_vwap[key][enddate]);
    final_series = pd.Series(final_list, index = stock_code); 
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_042(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(rank((vwap - close)) / rank((vwap + close)))
'''
#%%
def alpha042(enddate):
    temp1 = [];
    temp2 = [];
    final_list = [];
    for key,value in data_close.iteritems(): 
        temp1.append(data_vwap[key][enddate]-data_close[key][enddate]);
        temp2.append(data_vwap[key][enddate]+data_close[key][enddate]);
    rank_1st_list = Rank(temp1);
    rank_2nd_list = Rank(temp2);
    for i in range(len(rank_1st_list)):
        if (rank_2nd_list[i] == 0.0):
            final_list.append(float('nan'));
        else:
            final_list.append(rank_1st_list[i]/rank_2nd_list[i]);
    final_series = pd.Series(final_list, index = stock_code); 
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_044(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1* correlation(high, rank(volume), 5))
'''
#%%
def alpha044(enddate):
    #cpt 1st item
    rank_list = [[] for i in range(5)];
    rank_Series = [[] for i in range(5)];
    for day in range(5,0,-1):
        for key,value in data_volume.iteritems():
            if (data_volume[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_volume[key][enddate-day];
            rank_list[day-1].append(val);
        rank_list[day-1] = Rank(rank_list[day-1]);
        rank_Series[day-1] = pd.Series(rank_list[day-1], index = stock_code);
    frame1 = [rank_Series[i] for i in range(5)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    high_list = [[] for i in range(5)];
    high_Series = [[] for i in range(5)];
    for day in range(5,0,-1):
        for key,value in data_high.iteritems():
            if (data_high[key][enddate-day] == 0):
                val = float('nan');
            else:
                val = data_high[key][enddate-day];
            high_list[day-1].append(val);
        high_Series[day-1] = pd.Series(high_list[day-1], index = stock_code);
    frame1 = [high_Series[i] for i in range(5)];
    second_frame = pd.concat(frame1, axis=1);
    
    #cpt the corr
    corr_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        corr_list.append(-1*s1.corr(s2));
    final_series = pd.Series(corr_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%

#%%
'''
alpha_045(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1* ((rank((sum(delay(close, 5), 20) / 20))* correlation(close, volume, 2))* rank(correlation(sum(close, 5), sum(close, 20), 2))))
'''
#%%
def alpha045_part1(enddate):
    rank_list = [];
    for key,value in data_close.iteritems(): 
        temp = [data_close[key][enddate-5-i] for i in range(20)];
        rank_list.append(sum(temp)/20.0);
    final_list = Rank(rank_list);
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 

def alpha045_part2(enddate):
    #cpt 1st item
    vol_list = [[] for i in range(2)];
    vol_Series = [[] for i in range(2)];
    for day in range(2,0,-1):
        for key,value in data_volume.iteritems():
            val = data_volume[key][enddate-day];
            vol_list[day-1].append(val);
        vol_Series[day-1] = pd.Series(vol_list[day-1], index = stock_code);
    frame1 = [vol_Series[i] for i in range(2)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    close_list = [[] for i in range(2)];
    close_Series = [[] for i in range(2)];
    for day in range(2,0,-1):
        for key,value in data_high.iteritems():
            val = data_close[key][enddate-day];
            close_list[day-1].append(val);
        close_Series[day-1] = pd.Series(close_list[day-1], index = stock_code);
    frame1 = [close_Series[i] for i in range(2)];
    second_frame = pd.concat(frame1, axis=1);
    
    #cpt the corr
    corr_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        corr_list.append(s1.corr(s2));
    final_series = pd.Series(corr_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 

def alpha045_part3(enddate):
    #cpt 1st item
    close_list_1 = [[] for i in range(2)];
    close_Series_1 = [[] for i in range(2)];
    for day in range(2,0,-1):
        for key,value in data_close.iteritems():
            val = sum([data_close[key][enddate-day-i] for i in range(5)]);
            close_list_1[day-1].append(val);
        close_Series_1[day-1] = pd.Series(close_list_1[day-1], index = stock_code);
    frame1 = [close_Series_1[i] for i in range(2)];
    first_frame = pd.concat(frame1, axis=1);
    
    #cpt 2nd item
    close_list_2 = [[] for i in range(2)];
    close_Series_2 = [[] for i in range(2)];
    for day in range(2,0,-1):
        for key,value in data_high.iteritems():
            val = sum([data_close[key][enddate-day-i] for i in range(20)]);
            close_list_2[day-1].append(val);
        close_Series_2[day-1] = pd.Series(close_list_2[day-1], index = stock_code);
    frame1 = [close_Series_2[i] for i in range(2)];
    second_frame = pd.concat(frame1, axis=1);
    
    #cpt the corr
    corr_list = [];
    for i in range(len(stock_code)):
        s1 = first_frame.iloc[i][:];
        s2 = second_frame.iloc[i][:];
        corr_list.append(s1.corr(s2));
    final_series = pd.Series(corr_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;

def alpha045(enddate):
    first_list = list(alpha045_part1(enddate));
    second_list = list(alpha045_part2(enddate));
    third_list = list(alpha045_part3(enddate));
    
    final_list = [-first_list[i]*second_list[i]*third_list[i] for i in range(len(first_list))];
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series;
#%%
'''
alpha_053(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
(-1* delta((((close - low) - (high - close)) / (close - low)), 9))
'''
#%%
def alpha053(enddate):
    final_list = [];
    for key,value in data_close.iteritems():
        past_term1 = data_close[key][enddate - 9] - data_low[key][enddate - 9];
        past_term2 = data_high[key][enddate - 9] - data_close[key][enddate - 9];
        cur_term1 = data_close[key][enddate] - data_low[key][enddate];
        cur_term2 = data_high[key][enddate] - data_close[key][enddate];
        if ((past_term1 == 0.0) | (cur_term1 == 0.0)):
            final_list.append(float('nan'));
        else:
            past_val = (past_term1 - past_term2) / past_term1;
            cur_val = (cur_term1 - cur_term2) / cur_term1;
            dlt = cur_val - past_val;
            final_list.append(dlt);
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%
'''
alpha_054(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
((-1* ((low - close)* (open^5))) / ((low - high)* (close^5)))
'''
#%%
def alpha054(enddate):
    final_list = [];
    for key,value in data_close.iteritems():
        term1 = -(data_low[key][enddate]-data_close[key][enddate])*(data_open[key][enddate]**5);
        term2 = (data_low[key][enddate]-data_high[key][enddate])*(data_close[key][enddate]**5);
        if (term2 == 0.0):
            final_list.append(float('nan'));
        else:
            final_list.append(term1 / term2);
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%%
'''
alpha_101(enddate, index='all')
输入： 
enddate: 必选参数，计算哪一天的因子
index: 默认参数，股票指数，默认为所有股票’all’
输出： 
一个 Series：index 为成分股代码，values 为对应的因子值
因子公式： 
((close - open) / ((high - low) + .001))
'''
#%%
def alpha101(enddate):
    final_list = [];
    for key,value in data_close.iteritems():
        term1 = (data_close[key][enddate]-data_open[key][enddate]);
        term2 = (data_high[key][enddate]-data_low[key][enddate]+0.001);
        if (term2 == 0.0):
            final_list.append(float('nan'));
        else:
            final_list.append(term1 / term2);
    final_series = pd.Series(final_list, index = stock_code);
    final_series.sort_values(ascending=False, inplace=True);
    return final_series; 
#%% 
        


