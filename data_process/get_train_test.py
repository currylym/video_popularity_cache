# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import datetime
import warnings
from collections import OrderedDict
import sys
sys.path.append('..')

from parameters import MIN_POPULARITY,UPDATE_CYCLE,TRAIN_START_DATE,TRAIN_END_DATE,\
                        TEST_START_DATE,TEST_END_DATE,DATA_DIR,OUT_DIR
warnings.simplefilter(action='ignore', category=FutureWarning)

# 生成时间序列
def CreateVideoTimeseries(data_all,
                          start_date,
                          end_date,
                          freq=UPDATE_CYCLE,
                          min_popularity=MIN_POPULARITY
                          ):

    videos = list(OrderedDict.fromkeys(list(data_all['video_id']))) # 保持原来顺序

    #timeIndex
    data_timeindex = data_all[['video_id','time','user']].copy()
    data_timeindex['date'] = data_timeindex['time'].apply(lambda x:datetime.datetime.fromtimestamp(int(x)))
    data_timeindex = data_timeindex.sort_values(by = 'date',ascending = True)
    data_timeindex.set_index("date", inplace=True)

    timepoints = pd.date_range(start_date,end_date,freq = freq)
    #print(timepoints)
    v_timeSeries = pd.DataFrame(index = videos)
    index = 0
    for i in range(len(timepoints)-1):
        start = timepoints[i]
        end = (timepoints[i+1] - datetime.timedelta(hours=1)).strftime('%F %H')
        tmp = data_timeindex[start:end]
        tmp = dict(tmp.groupby('video_id').apply(len))
        v_timeSeries[timepoints[index]] = pd.Series(tmp,index = videos)
        index += 1
    v_timeSeries.fillna(0,inplace = True)
    #print(v_timeSeries.head())

    v_timeSeries = v_timeSeries[v_timeSeries.sum(axis = 1) > min_popularity]
    v_timeSeries = v_timeSeries.astype('int')
    videos1 = list(v_timeSeries.index)

    history_start = timepoints[0]
    history_end = (timepoints[-1] - datetime.timedelta(hours=1)).strftime('%F %H')
    history = data_timeindex[history_start:history_end]
    history = history[history['video_id'].apply(lambda x:x in videos1)]

    return v_timeSeries,history

#------------------------------read and process youku data-----------------------------------
# 读取youku原始数据
filepath = r'../%syoukudata.csv' % DATA_DIR
data = pd.read_csv(filepath)[['user','video_id','time']]
print('raw data shape:', data.shape)

# 生成训练集和历史记录
train_youku_ts,youku_history = CreateVideoTimeseries(data,TRAIN_START_DATE,TRAIN_END_DATE)
print('train timeseries shape:', train_youku_ts.shape)
print('youku history head:\n', youku_history.head())
print('youku history tail:\n', youku_history.tail())
print('[info]saving youku history data to ../%syouku_history.csv' % OUT_DIR)
youku_history.to_csv(r'../%syouku_history.csv' % OUT_DIR, index=False)
print('[info]saving youku train timeseries data to ../%strain_youku_ts.csv' % OUT_DIR)
train_youku_ts.to_csv(r'../%strain_youku_ts.csv' % OUT_DIR, index=True)

# 生成测试集和历史记录
test_youku_ts,_ = CreateVideoTimeseries(data,TEST_START_DATE,TEST_END_DATE)
print('test timeseries shape:', test_youku_ts.shape)
print('[info]saving youku test timeseries data to ../%stest_youku_ts.csv' % OUT_DIR)
test_youku_ts.to_csv(r'../%stest_youku_ts.csv' % OUT_DIR, index=True)

#------------------------------read and process iqiyi data-----------------------------------
# 读取iqiyi原始数据
filepath1 = r'../%siqiyidata.xlsx' % DATA_DIR
data1 = pd.read_excel(filepath1,sheetname=None);data1 = data1['Sheet1']
data1 = data1.rename(columns = {'uri_path':'video_id'})
print('iqiyi timeseries shape:', data1.shape)

# 生成iqiyi的时间序列的历史记录
train_iqiyi_ts,iqiyi_history = CreateVideoTimeseries(data1,TRAIN_START_DATE,TRAIN_END_DATE)
print('[info]saving iqiyi history data to ../%siqiyi_history.csv' % OUT_DIR)
iqiyi_history.to_csv(r'../%siqiyi_history.csv' % OUT_DIR, index=False)
print('[info]saving iqiyi timeseries data to ../%strain_test_iqiyi.csv' % OUT_DIR)
train_iqiyi_ts.to_csv(r'../%strain_iqiyi_ts.csv' % OUT_DIR, index=True)
