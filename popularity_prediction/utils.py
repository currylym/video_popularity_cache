# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import prettytable as pt

from sklearn.preprocessing import StandardScaler

# 同时打乱X和y
def together_shuffle(X,y):
    seed = np.random.randint(1000)
    np.random.seed(seed)
    per = np.random.permutation(y.shape[0])
    X_shuffle = {}
    for key in X:
        X_shuffle[key] = X[key][per,:]
    y_shuffle = y[per,]
    return X_shuffle,y_shuffle

#-----------------------------数据准备------------------------------------------

import sys
sys.path.append('..')
from parameters import HISTORY_NUM,SIMILAR_VIDEO_NUMS,OUT_DIR

# 准备流行度预测的数据
def prepare_data_for_pp(normalize=True,
                        shuffle=True,
                        remain_cold_start=True,
                        val_rate=0.3,
                        filter_all_zero_train=True,
                        filter_all_zero_test=True,
                        seed=1000):
    
    '''
    Params:
    ----------------------------------
        normalize:是否对数据进行归一化
        shuffle:是否打乱数据
        remain_cold_start:是否保留冷启动视频
        val_rate:验证集占训练集的比例
        filter_all_zero_train:去除x和y全0的训练样本
        filter_all_zero_test:去除x全零的测试样本
        seed:随机种子
    '''
    
    # 选择随机种子---划分训练集验证集的种子
    np.random.seed(seed)

    time_step = HISTORY_NUM
    similar_video_num = SIMILAR_VIDEO_NUMS

    f_train = open('../%strain.txt'%OUT_DIR, 'w')
    f_test = open('../%stest.txt'%OUT_DIR, 'w')
    f_val = open('../%sval.txt'%OUT_DIR, 'w')
    
    # 时序数据
    train_youku_ts = pd.read_csv(r'../%strain_youku_ts.csv' % OUT_DIR, index_col=0)
    test_youku_ts = pd.read_csv('../%stest_youku_ts.csv' % OUT_DIR, index_col=0)
    train_iqiyi_ts = pd.read_csv(r'../%strain_iqiyi_ts.csv' % OUT_DIR, index_col=0)
    # youku文本特征
    video_info_dict = json.loads(open(r'../%sfasttext_char_em.json' % OUT_DIR).read())
    # youku关系特征
    kg_embedding = json.load(open(r'../%skg_embedding/kge_res/entity_embedding.json' % OUT_DIR))
    # iqiyi相似视频
    similar_video = json.load(open(r'../%ssimilar_video.json' % OUT_DIR))
    print('youku train timeseries shape:', train_youku_ts.shape)
    print('youku test timeseries shape:', test_youku_ts.shape)

    x_train = {'main_data':[],
               'tags':[],
               'title':[],
               'description':[],
               'kg_embedding':[],
               'similar_iqiyi_video_ts':[]
               }
    y_train = []
    
    x_test = {'main_data':[],
              'tags':[],
              'title':[],
              'description':[],
              'kg_embedding':[],
              'similar_iqiyi_video_ts':[]
              }
    y_test = []

    x_val = {'main_data':[],
              'tags':[],
              'title':[],
              'description':[],
              'kg_embedding':[],
              'similar_iqiyi_video_ts':[]
              }
    y_val = []

    # 删除冷启动视频
    if not remain_cold_start:
        not_cold_start = list(set(train_youku_ts.index) & set(test_youku_ts.index))
        test_youku_ts = test_youku_ts.ix[not_cold_start,:]
        print('youku test timeseries shape after remove cold-start video:', test_youku_ts.shape)

    # 准备训练和验证数据
    for v in train_youku_ts.index:
        for i in range(train_youku_ts.shape[1]):
            if i+time_step < train_youku_ts.shape[1]:
                # 去除全0样本
                if filter_all_zero_train:
                    if sum(train_youku_ts.loc[v][i:i+time_step+1]) == 0:
                        continue
                # 每一条数据随机分到验证集和训练集
                if np.random.random() > val_rate:
                    x,y = x_train,y_train
                    f = f_train
                else:
                    x,y = x_val,y_val
                    f = f_val
                # 准备x
                x['main_data'].append(np.array(train_youku_ts.loc[v][i:i+time_step]))
                for key in ['tags','title','description']:
                    if v in video_info_dict:
                        x[key].append(np.array(video_info_dict[v][key]))
                    else:
                        x[key].append(np.zeros(300))
                x['kg_embedding'].append(np.array(kg_embedding[v]) if v in kg_embedding else np.zeros(100))
                if v in similar_video:
                    x['similar_iqiyi_video_ts'].append(
                        [np.array(train_iqiyi_ts.loc[j] [i:i + time_step]) for j in similar_video[v]]
                        + [np.zeros(time_step)]*(similar_video_num - len(similar_video[v])))
                else:
                    x['similar_iqiyi_video_ts'].append([np.zeros(time_step)]*similar_video_num)
                # 准备y
                y.append(train_youku_ts.loc[v][i+time_step])
                # 保存到中间文件
                line = '%s %s\t%s\n' % (v,' '.join(map(str,train_youku_ts.loc[v][i:i+time_step])),train_youku_ts.loc[v][i+time_step])
                f.write(line)

    # 准备测试数据
    for v in test_youku_ts.index:
        for i in range(test_youku_ts.shape[1]):
            if i+time_step < test_youku_ts.shape[1]:
                # 去除全0样本
                if filter_all_zero_test:
                    if sum(test_youku_ts.loc[v][i:i+time_step+1]) == 0:
                        continue
                # 准备x
                x_test['main_data'].append(np.array(test_youku_ts.loc[v][i:i+time_step]))
                for key in ['tags','title','description']:
                    if v in video_info_dict:
                        x_test[key].append(np.array(video_info_dict[v][key]))
                    else:
                        x_test[key].append(np.zeros(300))
                x_test['kg_embedding'].append(np.array(kg_embedding[v]) if v in kg_embedding else np.zeros(100))
                if v in similar_video:
                    x_test['similar_iqiyi_video_ts'].append(
                        [np.array(train_iqiyi_ts.loc[j] [i:i + time_step]) for j in similar_video[v]]
                        + [np.zeros(time_step)]*(similar_video_num - len(similar_video[v])))
                else:
                    x_test['similar_iqiyi_video_ts'].append([np.zeros(time_step)]*similar_video_num)
                # 准备y
                y_test.append(test_youku_ts.loc[v][i+time_step])
                # 保存到中间文件
                line = '%s %s\t%s\n' % (v,' '.join(map(str,test_youku_ts.loc[v][i:i+time_step])),test_youku_ts.loc[v][i+time_step])
                f_test.write(line)

    # 归一化
    if normalize:
        scalar = StandardScaler()
        scalar.fit(x_train['main_data'])
        x_train['main_data'] = scalar.transform(x_train['main_data'])
        x_test['main_data'] = scalar.transform(x_test['main_data'])
        x_val['main_data'] = scalar.transform(x_val['main_data'])
    # reshape
    x_train['main_data'] = np.reshape(x_train['main_data'],(len(x_train['main_data']),time_step,1))
    x_test['main_data'] = np.reshape(x_test['main_data'],(len(x_test['main_data']),time_step,1))
    x_val['main_data'] = np.reshape(x_val['main_data'],(len(x_val['main_data']),time_step,1))
    for key in ['tags','title','description','kg_embedding','similar_iqiyi_video_ts']:
        x_train[key] = np.array(x_train[key])
        x_test[key] = np.array(x_test[key])
        x_val[key] = np.array(x_val[key])
    y_train,y_test,y_val = np.array(y_train),np.array(y_test),np.array(y_val)
    # 打乱数据
    if shuffle:
        together_shuffle(x_train,y_train)
        together_shuffle(x_val,y_val)
        together_shuffle(x_test,y_test)

    print('data processed')
    
    return x_train,y_train,x_test,y_test,x_val,y_val

if __name__ == '__main__':
    prepare_data_for_pp()

#--------------------------缓存命中率相关函数-------------------------------------

def HitRate(y_test,y_pred,cache_size = 200):
    total_request = np.sum(y_test)
    tuple_list = list(zip(y_test,y_pred))
    tuple_list = sorted(tuple_list,key = lambda x:x[1])[::-1]

    cached_request = 0
    for i in range(min(len(y_test),cache_size)):
        cached_request += tuple_list[i][0]

    return cached_request/total_request

def plot1(results,mode='mean first',step=1):
    '''
    把不同方法的缓存命中率画在一个图里
    
    Params:
    -------
    results:字典格式。方法名：方法的流行度预测结果/结果列表（多次实验，考虑模型的不稳定性）
    mode:对多次实验结果的处理方式：
         'mean fisrt'指先对实验结果进行平均再计算缓存命中率
         'mean last'指先计算缓存命中率再对其进行平均
    step:画图时缓存大小的取值间隔
    '''
    y_test = results['y_test']
    video_num = len(y_test)
    cache_size_list = np.arange(0,video_num,step)
    cache_size_percent_list = cache_size_list/video_num
    def _get_hitRate(y_pred):
        res = []
        for cache_size in cache_size_list:
            res.append(HitRate(y_test,y_pred,cache_size))
        return res

    plt.figure()
    plt.xlim([0,0.2])
    plt.ylim([0.2,0.6])
    for label,y_pred in results.items():
        if isinstance(y_pred,list):
            if mode == 'mean last':
                res = [_get_hitRate(i) for i in y_pred]
                res = np.mean(res,axis=0)
            elif mode == 'mean first':
                y_pred = np.mean(y_pred,axis=0)
                res = _get_hitRate(y_pred)
            else:
                res = _get_hitRate(y_pred[mode])
        else:
            res = _get_hitRate(y_pred)
        plt.plot(cache_size_percent_list,res,label=label)
    plt.grid()
    plt.legend()
    
def print_hitRate_result_table(results,mode='mean first'):
    y_test = results['y_test']
    video_num = len(y_test)
    cache_size_percent_list = [0.05,0.1,0.15,0.2,0.25]
    cache_size_list = [int(i*video_num) for i in cache_size_percent_list]
    
    tb = pt.PrettyTable()
    tb.set_style(pt.PLAIN_COLUMNS)
    tb.field_names = ['method'] + list(map(lambda x:'hitRate@'+str(x),cache_size_percent_list))
    
    def _get_hitRate(y_pred):
        res = []
        for cache_size in cache_size_list:
            res.append(HitRate(y_test,y_pred,cache_size))
        return res
    for label,y_pred in results.items():
        if isinstance(y_pred,list):
            if mode == 'mean last':
                res = [_get_hitRate(i) for i in y_pred]
                res = np.mean(res,axis=0)
            elif mode == 'mean first':
                y_pred = np.mean(y_pred,axis=0)
                res = _get_hitRate(y_pred)
            else:
                res = _get_hitRate(y_pred[mode])
        else:
            res = _get_hitRate(y_pred)
        if label != 'y_test':
            tb.add_row([label] + list(map(lambda x:round(x,3),res)))
    print(tb)

def plot2(results):
    for i in results:
        if isinstance(results[i],list):
            results[i] = np.mean(results[i],axis=0)
    data = pd.DataFrame(results)
    data = data.sort_values(by=['y_test'],ascending=False)
    data.index = range(len(data))
    data[:100].plot()
    plt.savefig('fig1',dpi=800)

#---------------------------流行度预测相关函数------------------------------------
    
def MSE(results,mode='mean first'):
    def _mse(y_pred,y_test):
        return np.mean((y_pred-y_test)**2)
    MSE_res = {}
    for label,y_pred in results.items():
        y_test = results['y_test']
        if isinstance(y_pred,list):
            if mode == 'mean last':
                res = [_mse(y_pred,y_test) for i in y_pred]
                res = np.mean(res,axis=0)
                #print(label,mode)
            elif mode == 'mean first':
                y_pred = np.mean(y_pred,axis=0)
                res = _mse(y_pred,y_test)
                #print(label,mode)
        else:
            res = _mse(y_pred,y_test)
        if label != 'y_test':
            MSE_res[label] = res
    return MSE_res

def MAP(results,mode='mean first'):
    def _map(y_pred,y_test):
        return np.mean(abs(y_pred-y_test))
    MAP_res = {}
    for label,y_pred in results.items():
        y_test = results['y_test']
        if isinstance(y_pred,list):
            if mode == 'mean last':
                res = [_map(y_pred,y_test) for i in y_pred]
                res = np.mean(res,axis=0)
                #print(label,mode)
            elif mode == 'mean first':
                y_pred = np.mean(y_pred,axis=0)
                res = _map(y_pred,y_test)
                #print(label,mode)
        else:
            res = _map(y_pred,y_test)
        if label != 'y_test':
            MAP_res[label] = res
    return MAP_res

#打印不同方法下流行度预测的MSE和MAP指标
def plot_MSE_MAP_table(results,mode='mean first'):
    MSE_res = MSE(results,mode=mode)
    MAP_res = MAP(results,mode=mode)
    tb = pt.PrettyTable()
    tb.set_style(pt.PLAIN_COLUMNS)
    tb.field_names = ['method','mse','map']
    for method in MSE_res:
        row = [method,round(MSE_res[method],3),round(MAP_res[method],3)]
        tb.add_row(row)
    print(tb)
    return tb

#将结果保存到txt文件中
def save_result(tb,out_path):
    f = open(out_path,'w')
    f.write(str(tb))
    f.close()

#计算不同方法的mean/std/最优次数
def compare_method_in_N(n=10):

    def read(path):
        res = {}
        with open(path) as f:
            for line in f:
                line = line.strip().split(' ')
                line = [i for i in line if i]
                if line[0] == 'method':
                    continue
                res[line[0]] = line[1:]
        return res

    def _mean(nums):
        nums = list(map(float,nums))
        return np.mean(nums)

    def _std(nums):
        nums = list(map(float,nums))
        return np.var(nums)

    result = {}
    for i in range(1,n+1):
        path = 'result/2016-10-3_2016-10-23_2016-10-24_2016-10-30_default_%s.txt' % i
        res = read(path)
        for key in res:
            if key in result:
                result[key][0].append(res[key][0])
                result[key][1].append(res[key][1])
            else:
                result[key] = [[res[key][0]],[res[key][1]]]
    print(result)    
    df = pd.DataFrame(columns=['mse_mean','mse_std','map_mean','map_std','mse_max_time','map_max_time'])
    for key in result:
        df.loc[key,'mse_mean'] = _mean(result[key][0])
        df.loc[key,'mse_std'] = _std(result[key][0])
        df.loc[key,'map_mean'] = _mean(result[key][1])
        df.loc[key,'map_std'] = _std(result[key][1])
    
    return df
