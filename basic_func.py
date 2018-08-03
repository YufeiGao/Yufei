#%%
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import multiprocessing
import matplotlib.pyplot as plt
import scipy.io
import stats
#%%
'''
general idea:
1.load the table data_close as the stock price
2.code some basic alphas by using the simple functions
3.loop over the whole table, select one day as the enddate, return the portfolio
4.compare its IPO date with the enddate, if conflict, drop it
'''
#%%
def Rank(x):
    # x is a list, return the values equally distributed among 0 and 1
    len_x = len(x);
    step = 1.0/float(len_x - 1);
    cur_rank = []; ## record the rank for each entries on the list
    for i in x:
        rank = 1;
        for j in x:
            if (j < i):
                rank = rank + 1;
        cur_rank.append(rank);
    for idx,item in enumerate(cur_rank):
        cur_rank[idx] = (item - 1) * step;
    return cur_rank;

def Min(x, y):
    # x and y are lists, return the parallel min at each entries
    pmin = [];
    length = len(x);
    for i in range(length):
        min_xy = min(x[i], y[i]);
        pmin.append(min_xy);
    return pmin;

def Max(x, y):
    # x and y are lists, return the parallel min at each entries
    pmax = [];
    length = len(x);
    for i in range(length):
        max_xy = max(x[i], y[i]);
        pmax.append(max_xy);
    return pmax;

def StdDev(x, n):
    # return the std for the past n days in x
    # n must be less than 256
    return np.std(x[-n:]);

def sum_past(x,n):
    # sum the value for past n days
    return sum(x[-n:]);

def product_past(x,n):
    # get the product for past n days
    return np.prod(x[-n:]);

def Correlation(x,y,n):
    # return the corr btw x and y in past n days
    # n must be less than 256
    return np.corrcoef(x[-n:], y[-n:]);

def Tail(x, lower, upper, newval):
    # set the values of x to newval if they are btw lower and upper
    len_x = len(x);
    for i in range(len_x):
        if (x[i] > lower) & (x[i] < upper):
            x[i] = newval;
    return x;

def Ts_Min(x,n):
    return min(x[-n:]);

def Ts_Max(x,n):
    return max(x[-n:]);

def SignedPower(x,e):
    result = [];
    for i in range(len(x)):
        val = np.sign(x[i])*(abs(x[i])**e);
        result.append(val);
    return (result);

def Ts_Rank(x,n):
    # rank the last n days' data
    result = Rank(x[-n:], n);
    return (result);

def Ts_skewness(x,n):
    return stats.skewness(x[-n:]);

def Ts_Kurtosis(x,n):
    return stats.kurtosis(x[-n:]);

def Pasteurize(x):
    for i in range(len(x)):
        if (x[i] == float("inf")) | (x[i] == float("-inf")):
            x[i] = float("nan");
    return (x);

def delay(x,n):
    return x[-n];

def delta(x,n):
    return x[-1] - x[-n];

def scale(x,a = 1):
    sum_x = 0; #scale x with abs sum to a
    for i in x:
        sum_x = sum_x + abs(i);
    unit = a / float(sum_x);
    for idx in range(len(x)):
        x[idx] = (x[idx] * unit);
    return x;

def decay_linear(x,n):
    weight = []; #assign weight from n days before, the farest the most
    for i in range(n,0,-1):
        weight.append(i); 
    weight_adj = scale(weight);
    sum_x = 0;
    for i in range(n):
        val = weight_adj[i]*x[-(i+1)];    #weightd avg
        sum_x = sum_x+val;
    return sum_x;

def Ts_argmin(x,n):
    # on which day the min val will get
    target = Ts_Min(x,n);
    day = -1;
    for i in range(n,0,-1):
        if (x[-i] == target):
            day = (n-i+1);
            break;
    return day;

def Ts_argmax(x,n):
    # on which day the max val will get
    target = Ts_Max(x,n);
    day = -1;
    for i in range(n,0,-1):
        if (x[-i] == target):
            day = (n-i+1);
            break;
    return day;
#%%