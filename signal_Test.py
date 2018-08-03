'''
next step's work:
Translate date to integer, eg: 2001/01/01 -> 1
Add stock's name matches with code
drop stocks whose turnover is 0 (mathes with code)
drop stocks whose IPO data is not before current date 
'''
import os
os.chdir('D:\\gyf\\alpha101')
import pandas as pd
from alpha_func import *
from regression import *
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import multiprocessing
import matplotlib.pyplot as plt
import scipy.io
#%%
# Date Translation to int
mat = scipy.io.loadmat('C:/Users/HP/Desktop/data_gp.mat')
data_all = pd.DataFrame(mat['AllAStockSet'])
data_all.head(20)
file1 = pd.read_csv('C:\\Users\HP\Desktop\TradingDay.csv')
trading_date = pd.DataFrame(file1)
trading_date.columns = ['number','date']
trading_date.drop(['number'],axis=1,inplace=True)
def date_to_int(enddate):
    val = 0;
    for i in range(len(trading_date)):
        if (trading_date['date'][i][2:-2] == enddate):
            val = i;
            break;
    return val;
#%%
file4 = pd.read_csv('C://Users/HP/Desktop/pe_d.csv');
data_pe = pd.DataFrame(file4);
data_pe.fillna(0, inplace=True);
data_pe.set_index(['Date'],inplace=True);
#%%
file5 = pd.read_csv('C://Users/HP/Desktop/StockPool.csv',encoding='gbk');
data_ipo = pd.DataFrame(file5);
data_ipo.set_index(['codes'],inplace=True);
#%%
# Add Stock's name
data_name = [(data_all.iloc[2])[i][0] for i in range(len(data_all.columns))];
stock_code = [(data_all.iloc[1])[i][0] for i in range(len(data_all.columns))];
data_name = pd.Series(data_name, index = stock_code);
data_name.head(20);
#%%
# Add Turnover
file2 = pd.read_csv('C://Users/HP/Desktop/turn_d.csv');
data_turnover = pd.DataFrame(file2);
data_turnover.fillna(0, inplace=True);
data_turnover.set_index(['Date'],inplace=True);
data_turnover.head(20)
'''
Notice that universe in table 'Turnover' is less than that in 'data'
'''
def add_turnover(enddate,data_add_name):
    turn_list = [];
    for code in stock_code:
        idx = list(data_turnover.columns)
        if (code not in idx):
            turn_list.append(0.0);
        else:
            turn_list.append(data_turnover.loc[enddate][code]); 
    data_turn = pd.Series(turn_list, index = stock_code);
    frame = [data_add_name, data_turn];
    data_add_all = pd.concat(frame, axis=1);
    data_add_all.columns = ['Name','Turnover'];
    return data_add_all;

#%%
#drop zero turnover
def drop_zero_turn(enddate, stock_series, turn_table, n = len(stock_code)):      
    for key in stock_series.index:
        if (turn_table['Turnover'][key] == 0.0):
            stock_series.drop([key],inplace=True);
    return stock_series[:n];
#%%
#drop PE <=0 or >= 100
def select_pe(enddate, stock_series, n = len(stock_code)):
    for key in stock_series.index:
        if ((key not in data_pe.columns)|(data_pe[key][enddate] <= 0.0)):
            stock_series.drop([key],inplace=True);
    return stock_series[:n];
#%%
# drop the stocks less than 30 days within its IPO
import datetime
def select_ipo(enddate, stock_series, n = len(stock_code)):
    for key in stock_series.index:
        if (key not in data_ipo.index):
            stock_series.drop([key],inplace=True);
        else:
            ipo_day = data_ipo['ipodate'][key];
            d1 = datetime.datetime.strptime(ipo_day, '%Y/%m/%d');
            dlt = datetime.timedelta(days = 90);
            cur_time = datetime.datetime.strptime(enddate, '%Y/%m/%d');
            d2 = d1 + dlt;
            if (cur_time < d2):
                stock_series.drop([key],inplace=True);
    return stock_series[:n];     
#%%
# signal selection(top ranked n stocks)
def signal_select(enddate, alpha_list, n = len(stock_code)):
    wdate = date_to_int(enddate);
    data_add_turn = add_turnover(enddate,data_name);
    sigList=[];
    for fptr in alpha_list:
        nameList=[]
        sig = fptr(wdate);
        sig_temp1 = select_ipo(enddate, sig, n);
        sig_temp2 = drop_zero_turn(enddate, sig_temp1, data_add_turn, n);
        sig_mod = select_pe(enddate, sig_temp2, n);
        for key in sig_mod.index:
            nameList.append(data_add_turn['Name'][key]);
        nameList = pd.Series(nameList, index = sig_mod.index)
        sigList.append(sig_mod);
    return sigList[:n], nameList;
#%%
file3 = pd.read_csv('C:\\Users\HP\Desktop\pct_d.csv');
data_pct = pd.DataFrame(file3);
data_pct.set_index(['Date'],inplace=True);
data_pct.head()
#%%
# Add pct
# return a series with pct on enddate
def get_pct(enddate,sig_list):
    pct_list = [];
    for code in sig_list.index:
        if (code not in (list(data_pct.columns))):
            pct_list.append(0.0);
        else:
            pct_list.append(data_pct[code][enddate]); 
    pct_series = pd.Series(pct_list, index = (sig_list.index));
    return pct_series;
#%%
def back_test(startdate, stock_pool, n):
    diff_date = date_to_int('2005/1/4'); # the init date on pct table is '2005/1/4'
    startdate = date_to_int(startdate);
    his_pct = []; # n sublist, each contain KEY-many pct
    for i in range(1,n+1):
        temp = [];
        cur_pct = get_pct(startdate+i, stock_pool);
        for key in stock_pool.index:
            temp.append(cur_pct[key]);
        his_pct.append(temp);
    # cpt port's pct on this day
    daily_port_pct = [sum(his_pct[i])/len(his_pct[i]) for i in range(len(his_pct))];
    port_pct = 1.0;
    for i in range(len(daily_port_pct)):
        port_pct = port_pct*(1 + daily_port_pct[i]/100.0);
    port_pct = (port_pct-1)*100;   
    return daily_port_pct, port_pct;
#%%
def int_to_date(enddate):
    return trading_date['date'][enddate][2:-2];   
#%%
#%%
def benchmark_pct(startdate, no_period, n):
    bmk_pct = data_pct['000300.SH'];
    wdate = date_to_int(startdate);
    diff_date = date_to_int('2005/1/4');
    wdate = wdate - diff_date;
    port_pct = [];
    daily_port_pct = [[] for i in range(no_period)]; #contain no_period sublist 
    for j in range(no_period):
        daily_pct = [];
        for i in range(1,n+1):
            daily_pct.append(1 + bmk_pct[wdate+i]/100.0);
            daily_port_pct[j].append(bmk_pct[wdate+i]);
        cur_pct = np.prod(daily_pct);
        port_pct.append((cur_pct-1)*100);
        wdate = wdate + n;
    return daily_port_pct, port_pct;
#%%
def back_test_cont(startdate, alpha_list, no_period, no_stock, n, fee = 0.0):
    wdate = date_to_int(startdate);
    daily_bmk_pct, bmk_pct = benchmark_pct(startdate, no_period, n);
    diff_date = date_to_int('2005/1/4');
    wdate = wdate - diff_date;
    daily_net_pct = [[] for i in range(no_period)];
    daily_total_pct = [[] for i in range(no_period)];
    for i in range(no_period): 
        print ('Date: ', wdate);
        result, result_with_name = signal_select(int_to_date(wdate+diff_date), alpha_list);
        daily_cur_pct, cur_pct = back_test(int_to_date(wdate),result[0][:no_stock],n);
        for j in range(n):
            daily_net_pct[i].append(daily_cur_pct[j]-daily_bmk_pct[i][j]);
            daily_total_pct[i].append(daily_cur_pct[j]);
        wdate = wdate + n;
    return daily_total_pct, daily_net_pct, result_with_name[:no_stock];
#%%
daily_port_pct, daily_port_net_pct, result_with_name = back_test_cont('2010/1/4', [volume_factor], 192, 100, 10)
#%%
def get_pnl(daily_port_pct):
    pnl_list = [];
    cur_pnl = 1.0;
    for i in range(len(daily_port_pct)):
        for j in range(len(daily_port_pct[i])):
            cur_pnl = cur_pnl*(1+daily_port_pct[i][j]/100.0);
            pnl_list.append(cur_pnl);
    return pnl_list;
#%%
import datetime
b = benchmark_pct('2010/1/4', 192, 10)
bmk_list = get_pnl(b[0])
pnl_list_total = get_pnl(daily_port_pct)
pnl_list_net = get_pnl(daily_port_net_pct)
startdate = '2010/1/4'
n = 10
no_period = 192
plt.figure(figsize = (10, 5))
plt.grid(True)
time_list = []
for i in range(1, n*no_period+1):
    cur_time = int_to_date(date_to_int(startdate)+i);
    pos = cur_time.split('/');
    time_list.append(datetime.datetime(int(pos[0]),int(pos[1]),int(pos[2])));
plt.plot(time_list, bmk_list, color = 'blue')
plt.plot(time_list, pnl_list_total, color = 'orange')
plt.plot(time_list, pnl_list_net, color = 'green')
plt.xlabel('timeline')
plt.ylabel('pnl at this period')
plt.title('volume_factor from ' + startdate)
plt.show()
#%%
def turnover_btw_alpha(startdate, alpha_list):
    result = signal_select(startdate, alpha_list);
    result_a = result[0][:100];
    result_b = result[1][:100];
    no_same_idx = 0;
    for key in result_a.index:
        if key in result_b.index:
            no_same_idx = no_same_idx + 1;
    return (no_same_idx / 100.0);
turnover_btw_alpha('2018/1/9', [alpha018, alpha012])
#%%
def turnover_btw_date(startdate, enddate, alpha_list):
    result_1 = signal_select(startdate, alpha_list);
    result_2 = signal_select(enddate, alpha_list);
    result_a = result_1[0][:100];
    result_b = result_2[0][:100];
    no_same_idx = 0;
    for key in result_a.index:
        if key in result_b.index:
            no_same_idx = no_same_idx + 1;
    return no_same_idx/100.0;
turnover_btw_date('2008/1/2', '2008/1/14', [alpha005])
#%%
def cpt_drawback(test_data):
    # min position
    idx_j = np.argmax(np.maximum.accumulate(test_data)-test_data);
    # max position
    idx_i = np.argmax(test_data[:idx_j]);
    max_drawback = (test_data[idx_j]-test_data[idx_i])/test_data[idx_i]*100;
    plt.plot(test_data);
    plt.plot([idx_i, idx_j], [test_data[idx_i], test_data[idx_j]], 'o', color="r")
    return max_drawback;
#%%
test_data = get_pnl(daily_port_net_pct);
max_draw = cpt_drawback(test_data);
max_draw
#%%
port_data = [];
b = benchmark_pct('2017/1/3', 24, 10);
for i in range(len(daily_port_pct)):
    for j in daily_port_pct[i]:
        port_data.append(j);
port_pnl = get_pnl(daily_port_pct);
bmk_pnl = get_pnl(b[0]);
IR = cpt_IR(bmk_pnl, port_pnl, port_data);
IR
#%%
SR = cpt_SR(port_pnl, port_data);
SR
#%%
pct_now = (port_pnl[-1] - port_pnl[0])/port_pnl[0]*100.0;
pct_now
#%%
port_net_pnl = get_pnl(daily_port_net_pct);
pct_net_now = (port_net_pnl[-1] - port_net_pnl[0])/port_net_pnl[0]*100.0;
pct_net_now
#%%
def cpt_IR(bmk_pnl, port_pnl, port_data):
    excess_ret = []
    for month in range(12):
        # cpt the return for the bmk:
        bmk_ret = (bmk_pnl[(month+1)*20-1] - bmk_pnl[month*20])/bmk_pnl[month*20]*100;
        # cpt the return for the portfolio:
        port_ret = (port_pnl[(month+1)*20-1] - port_pnl[month*20])/port_pnl[month*20]*100;
        # cpt the excess return
        excess_ret.append(port_ret - bmk_ret);
    Avg = np.average(excess_ret);
    Std = np.std(excess_ret);
    return (Avg/Std)*np.sqrt(12);
def cpt_SR(port_pnl, port_data):
    excess_ret = []
    for month in range(12):
        # cpt the return for the portfolio:
        port_ret = (port_pnl[(month+1)*20-1] - port_pnl[month*20])/port_pnl[month*20]*100;
        # cpt the excess return
        excess_ret.append(port_ret - 0.0);
    Avg = np.average(excess_ret);
    Std = np.std(excess_ret);
    return (Avg/Std)*np.sqrt(12);
#%%
# cpt the winrate after one period
def cpt_winrate(startdate, alpha_list, time_len):
    # get the signal list
    sig_list, nameList = signal_select(startdate, alpha_list);
    # get the bmk pct
    bmk, bmk_pct = benchmark_pct(startdate, 1, time_len);
    wdate = date_to_int(startdate);
    diff_date = date_to_int('2005/1/4');
    wdate = wdate - diff_date;
    win_loss = [1 for i in range(100)];
    for day in range(time_len): # loop over the whole horizontal    
        # get the stock pct
        stock_ret = get_pct(wdate + day + 1, sig_list[0][:100]);
        # count the win rate
        win_loss = [win_loss[i]*(1 + stock_ret[i]/100.0) for i in range(len(stock_ret))];
        
    win_loss_sign = [np.sign((win_loss[i]-1.0)-bmk_pct[0]/100.0) for i in range(len(win_loss))];
    no_win = win_loss_sign.count(1);
    winrate = no_win/float(len(win_loss));
    return winrate;
#%%
winrate = [];
for period in range(24):
    startdate = int_to_date(date_to_int('2017/1/3')+period*10);
    temp = cpt_winrate(startdate, [volume_factor], 10);
    winrate.append(temp);
avg_winrate = np.average(winrate)
#%%
import WindPy


    
    