# -*- coding: utf-8 -*-

'''
一些常用函数
'''
import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.append('..')
import pickle
import random
import prettytable as pt
from keras import backend as K
from keras.models import load_model,Model

from config import MAIN_PATH
from parameters import HISTORY_NUM,UPDATE_CYCLE,SIMILAR_VIDEO_NUMS
from popularity_prediction2.utils import prepare_data_for_pp
from main_model import RankModel

#------------------------------------------------------------------------------

def save(data,path):
    fout = open(path, 'wb')
    pickle.dump(data, fout)
    fout.close()
    
def load(path):
    fin = open(path, 'rb')
    data = pickle.load(fin)
    fin.close()
    return data

def SaveModel(model,path):
    model.save_weights(path)

def LoadModel(path):
    #model = RankModel()
    model = RankModel(input_shape=(HISTORY_NUM,1),
                      lstm_dim=16,
                      filters=8,
                      kernel_size=3,
                      residual=False)
    model.load_weights(path)
    return model

def save_data_local(fdir=MAIN_PATH+'/out/processed_data/'):
    x_train,y_train,x_test,y_test,x_val,y_val = prepare_data_for_pp()
    if not os.path.exists(fdir):
        os.mkdir(fdir)
    save(x_train,fdir+'x_train.pkl')
    save(y_train,fdir+'y_train.pkl')
    save(x_test,fdir+'x_test.pkl')
    save(y_test,fdir+'y_test.pkl')
    save(x_val,fdir+'x_val.pkl')
    save(y_val,fdir+'y_val.pkl')
    print('data save into dir : %s' % fdir)

#------------------------------------------------------------------------------
    
def HitRate(y_test,y_pred,cache_size=200):
    total_request = np.sum(y_test)
    tuple_list = list(zip(y_test,y_pred))
    tuple_list = sorted(tuple_list,key = lambda x:x[1])[::-1]

    cached_request = 0
    for i in range(min(len(y_test),cache_size)):
        cached_request += tuple_list[i][0]

    return cached_request/total_request

def print_hitRate_result_table(y_test,y_pred,label,mode='mean first'):
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
    if isinstance(y_pred,list):
        if mode == 'mean last':
            res = [_get_hitRate(i) for i in y_pred]
            res = np.mean(res,axis=0)
        elif mode == 'mean first':
            y_pred = np.mean(y_pred,axis=0)
            res = _get_hitRate(y_pred)
    else:
        res = _get_hitRate(y_pred)
    tb.add_row([label] + list(map(lambda x:round(x,3),res)))
    print(tb)

#------------------------------------------------------------------------------

def get_all_pairs(sample=True,split=0.1):
    '''
    在历史数据上产生所有pair对数据。具体逻辑是在同一时刻将所有正负样本进行组合。
    
    params:
    -------
    sample:是否采样获取数据。如测试集的数据是18点-24点，则所有训练集/验证集的pair对都是在这个时刻抽取的。
    split:训练集和测试集划分比例。
    '''
    time_step = HISTORY_NUM
    youku_timeseries = pd.read_csv(MAIN_PATH+'/out/train_test_youku.csv',index_col = 0)
    #为了保证每个数据的label是相同时刻的，因此数据需要进行采样，sample_f为采样周期
    if sample:
        sample_f = int(24/int(UPDATE_CYCLE[:-1]))
    else:
        sample_f = 1
    
    x_all = []
    for i in range(time_step,youku_timeseries.shape[1]-1,sample_f):
        y = youku_timeseries.iloc[:,i]
        y_pos = list(y[y>0].index)
        y_neg = list(y[y==0].index)
        pairs = [(pos,neg,i-time_step,i) for pos in y_pos for neg in y_neg]
        x_all.extend(pairs)
    
    split_index = int(len(x_all)*split)
    x_train = x_all[split_index:]
    x_val = x_all[:split_index]
    
    print('data processed')
    print('train num:%d' % (len(x_train)))
    print('test num:%d' % (len(x_val)))
    return x_train,x_val

def get_data_num(flag='train'):
    x_train,x_val = get_all_pairs(sample=True,split=0.1)
    if flag == 'train':
        x = x_train
    elif flag == 'val':
        x = x_val
    return len(x)

def data_generator_pairwise(flag='train',batch_size=512):
    youku_timeseries = pd.read_csv(MAIN_PATH+'/out/train_test_youku.csv',index_col = 0)
    iqiyi_timeseries = pd.read_csv(MAIN_PATH+'/out/train_test_iqiyi.csv',index_col = 0)
    video_info_dict = load(MAIN_PATH+'/out/word_char_em.pkl')
    #kg_embedding = json.load(open(MAIN_PATH+'/out/kg_embedding/kge_res/entity_embedding.json'))
    similar_video = json.load(open(MAIN_PATH+'/out/similar_video.json'))
    
    x_train,x_val = get_all_pairs(sample=True,split=0.1)
    if flag == 'train':
        x = x_train
    elif flag == 'val':
        x = x_val
    random.shuffle(x)
    
    def get_related_series(youku_vid,start,end):
        #对与优酷视频youku_vid相似的iqiyi视频抽取同一时刻的time series
        res = []
        for iqiyi_vid in similar_video[youku_vid]:
            res.append(iqiyi_timeseries.loc[iqiyi_vid][start:end].tolist())
        if len(res) < SIMILAR_VIDEO_NUMS:
            res.extend([[0]*HISTORY_NUM]*(SIMILAR_VIDEO_NUMS-len(res)))
        return np.array(res)
    
    index = 0
    #处理信息缺失的视频
    NA = {'title':np.zeros((8,500),dtype='float'),
          'tags':np.zeros((4,500),dtype='float'),
          'description':np.zeros((30,500),dtype='float')}
    while True:
        if index <= len(x)//batch_size:
            x_batch = x[index*batch_size:(index+1)*batch_size]
            batch_dict = [[] for _ in range(10)]
            for i in x_batch:
                #print(i)
                v_pos,v_neg,start,end = i
                batch_dict[0].append(video_info_dict.get(v_pos,NA)['title'])
                batch_dict[1].append(video_info_dict.get(v_pos,NA)['tags'])
                batch_dict[2].append(video_info_dict.get(v_pos,NA)['description'])
                batch_dict[3].append(youku_timeseries.loc[v_pos][start:end].tolist())
                batch_dict[4].append(get_related_series(v_pos,start,end))
                batch_dict[5].append(video_info_dict.get(v_neg,NA)['title'])
                batch_dict[6].append(video_info_dict.get(v_neg,NA)['tags'])
                batch_dict[7].append(video_info_dict.get(v_neg,NA)['description'])
                batch_dict[8].append(youku_timeseries.loc[v_neg][start:end].tolist())
                batch_dict[9].append(get_related_series(v_neg,start,end))
            batch_dict = [np.array(i) for i in batch_dict]
            #输入需要是三维的
            #第一维不能写作batch_size，最后一个batch样本数目一般较少
            batch_dict[3] = np.reshape(batch_dict[3],(-1,HISTORY_NUM,1))
            batch_dict[8] = np.reshape(batch_dict[8],(-1,HISTORY_NUM,1))
            #for i in batch_dict:
                #print(i.shape)
            keys = [i+j for i in ['pos_','neg_'] for j in ['title','tags','description',
                    'history_viewing','related_series']]
            yield dict(zip(keys,batch_dict)),np.zeros((len(x_batch),2),dtype='float')
            index += 1
        else:
            index = 0
            random.shuffle(x)

#得到训练数据并注入特征
def get_testdata():
    
    youku_timeseries = pd.read_csv(MAIN_PATH+'/out/train_test_youku.csv',index_col = 0)
    iqiyi_timeseries = pd.read_csv(MAIN_PATH+'/out/train_test_iqiyi.csv',index_col = 0)
    video_info_dict = load(MAIN_PATH+'/out/word_char_em.pkl')
    #kg_embedding = json.load(open(MAIN_PATH+'/out/kg_embedding/kge_res/entity_embedding.json'))
    similar_video = json.load(open(MAIN_PATH+'/out/similar_video.json'))
    
    x_test = {'title':[],
              'tags':[],
              'description':[],
              'history_viewing':[],
              'related_series':[]}
    y_test = np.array(youku_timeseries.iloc[:,-1])
    
    def get_related_series(youku_vid,start,end):
        #对与优酷视频youku_vid相似的iqiyi视频抽取同一时刻的time series
        res = []
        for iqiyi_vid in similar_video[youku_vid]:
            res.append(iqiyi_timeseries.loc[iqiyi_vid][start:end].tolist())
        if len(res) < SIMILAR_VIDEO_NUMS:
            res.extend([[0]*HISTORY_NUM]*(SIMILAR_VIDEO_NUMS-len(res)))
        return np.array(res)
    NA = {'title':np.zeros((8,500),dtype='float'),
          'tags':np.zeros((4,500),dtype='float'),
          'description':np.zeros((30,500),dtype='float')}
    
    x_test['history_viewing'] = np.reshape(np.array(youku_timeseries.iloc[:,-8:-1]),(-1,HISTORY_NUM,1))
    for youku_vid in youku_timeseries.index:
        for key in ['title','tags','description']:
            x_test[key].append(video_info_dict.get(youku_vid,NA)[key])
        x_test['related_series'].append(get_related_series(youku_vid,-8,-1))
    x_test = {key:np.array(value) for key,value in x_test.items()}
    
    return x_test,y_test

#------------------------------------------------------------------------------

def train_model(batch_size=512,epochs=10,**kwargs):
    train_data_num = get_data_num(flag='train')
    val_data_num = get_data_num(flag='val')
    model = RankModel(input_shape=(HISTORY_NUM,1),
                      lstm_dim=16,
                      filters=8,
                      kernel_size=3,
                      residual=False,
                      **kwargs)
    model.fit_generator(generator=data_generator_pairwise(flag='train',batch_size=batch_size),
                        steps_per_epoch=train_data_num//batch_size+1,
                        epochs=epochs,
                        validation_data=data_generator_pairwise(flag='val',batch_size=batch_size),
                        validation_steps=val_data_num//batch_size+1)
    SaveModel(model,path='rank_model_weights.h5')   


    
#使用rank模型对测试集作流行度预测 
def predict(model_path='rank_model_weights.h5'):
    #导入模型并选取中间层
    model = LoadModel(model_path)
    #model.summary()
    intermediate_layer_model = Model(inputs=model.input[:5],
                                 outputs=model.layers[-2].get_output_at(0))
    intermediate_layer_model.summary()
    #读取测试数据，并进行键值映射
    x_test,_ = get_testdata()
    x_test = {'pos_'+key:value for key,value in x_test.items()}
    y_pred = intermediate_layer_model.predict(x_test)
    return y_pred

if __name__ == '__main__':
    save_data_local(fdir=MAIN_PATH+'/out/processed_data/')
    y_rank_pred = predict(model_path='rank_model_weights.h5')
    _,y_test = get_testdata()
    print_hitRate_result_table(y_test,y_rank_pred,label='rank',mode='mean first')
    