# -*- coding: utf-8 -*-

'''
产生测试seq2seq模型的生成数据
'''

import numpy as np
import random
from itertools import product
from keras.utils import to_categorical

def gen_add_data(data_num,in_len=10,out_len=5,max_single_num=1000):
    
    alphas = ['0','1','2','3','4','5','6','7','8','9','+','-','<pad>']
    alphas_map = dict(zip(alphas,range(len(alphas))))
    
    def _num_to_seq(num):
        return list(str(num))
    
    def _pad_onehot(seq,max_len):
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + ['<pad>']*(max_len-len(seq))
        seq = [alphas_map[i] for i in seq]
        return to_categorical(seq)
    
    def _str_operation(a,b,ops):
        if ops == '+':
            return a+b
        elif ops == '-':
            return a-b
        else:
            print('ops error!')
    
    def _gen_data():
        #产生不重复的数据
        a_list = np.arange(1,max_single_num).tolist()
        b_list = np.arange(1,max_single_num).tolist()
        ops_list = ['+','-']
        all_combines = list(product(a_list,b_list,ops_list))
        random.shuffle(all_combines)
        data = all_combines[:data_num]
        for a,b,ops in data:
            x = _num_to_seq(a) + [ops] + _num_to_seq(b)
            y = _num_to_seq(_str_operation(a,b,ops))
            #print(a,ops,b,'=',_str_operation(a,b,ops))
            yield _pad_onehot(x,in_len),_pad_onehot(y,out_len)
    
    X = []
    Y = []
    for x,y in _gen_data():
        X.append(x)
        Y.append(y)
    return np.array(X),np.array(Y) 

def add_data_decoder(X,Y,Y1):
    #对X，Y，Y1数据进行解码，分别代表表达式，准确结果和预测结果，并打印
    alphas = ['0','1','2','3','4','5','6','7','8','9','+','-','<pad>']
    alphas_map_r = dict(zip(range(len(alphas)),alphas))
    
    def _onehot_to_num(matrix):
        l = np.argmax(matrix,axis=1)
        l = [alphas_map_r[i] for i in l]
        l = [i for i in l if i != '<pad>']
        return ''.join(l)
    
    #计算预测值和真实值的平均偏差
    mean_bias = 0
    for x,y,y1 in zip(X,Y,Y1):
        x_,y_,y1_ = _onehot_to_num(x),_onehot_to_num(y),_onehot_to_num(y1)
        bias = abs(int(y1_) - int(y_))
        mean_bias += bias
        print('expression:%s right answer:%s predicted answer:%s bias:%d' % (x_,y_,y1_,bias))
    mean_bias = mean_bias/X.shape[0]
    print('mean bias:%.2f' % mean_bias)

if __name__ == '__main__':
    X,Y = gen_add_data(data_num=10,in_len=10,out_len=5,max_single_num=5)
    add_data_decoder(X,Y,Y)