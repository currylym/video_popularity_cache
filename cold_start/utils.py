# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
print(sys.path)

from config import MAIN_PATH
youku_series_path = MAIN_PATH+'/out/train_test_youku.csv'
iqiyi_series_path = MAIN_PATH+'/out/train_test_iqiyi.csv'
emb_path = MAIN_PATH+'/out/word_char_em.pkl'
from parameters import HISTORY_NUM

import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def load(path):
    fin = open(path, 'rb')
    data = pickle.load(fin)
    fin.close()
    return data

#准备冷启动数据
def prepare_data():
    res = []
    youku_series = pd.read_csv(youku_series_path,index_col=0)
    for v in youku_series.index:
        for j in range(youku_series.shape[1]):
            if j+HISTORY_NUM > youku_series.shape[1] - 2:continue
            if youku_series.loc[v][j:j+HISTORY_NUM].tolist() == [0]*HISTORY_NUM:
                res.append(['youku',v,j,j+HISTORY_NUM,youku_series.loc[v][j+HISTORY_NUM]])
    '''
    iqiyi_series = pd.read_csv(iqiyi_series_path,index_col=0)
    for v in iqiyi_series.index:
        for j in range(iqiyi_series.shape[1]):
            if j+HISTORY_NUM > iqiyi_series.shape[1] - 2:continue
            if iqiyi_series.loc[v][j:j+HISTORY_NUM].tolist() == [0]*HISTORY_NUM:
                res.append(['iqiyi',v,j,j+HISTORY_NUM,iqiyi_series.loc[v][j+HISTORY_NUM]])
    '''
    return res
    #print(res[:100])

def Pooling(array2D,mode='max'):
    if mode == 'max':
        return np.max(array2D,axis=0)
    if mode == 'mean':
        return np.mean(array2D,axis=1)

#查看冷启动数据的可区分性
def plot_data_2D(start=10,use_feature='title',mode='max'):
    embedding = load(emb_path)
    print('embedding read in!')
    data = prepare_data()
    data_filter = [i for i in data if i[2] == start]
    print(len(data_filter))
    X = [Pooling(embedding[i[1]][use_feature],mode=mode) for i in data_filter if i[1] in embedding]
    Y = [i[-1] for i in data_filter if i[1] in embedding]
    assert len(X) == len(Y)
    
    X_transform = TSNE(n_components=2).fit_transform(np.array(X))
    X_positive = [x for x,y in zip(X_transform,Y) if y == 1]
    X_negative = [x for x,y in zip(X_transform,Y) if y == 0]
    
    plt.figure()
    for pos in X_positive:
        plt.scatter(*pos,color='red')
    for neg in X_negative:
        plt.scatter(*neg,color='yellow')
    plt.title('start=%d use=%s mode=%s' % (start,use_feature,mode))
    plt.savefig('start=%d use=%s mode=%s' % (start,use_feature,mode),dpi=800)   

if __name__ == '__main__':
    prepare_data()
    plot_data_2D(start=20,use_feature='tags',mode='mean')