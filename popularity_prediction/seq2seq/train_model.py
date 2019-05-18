# -*- coding: utf-8 -*-

import numpy as np
from seq2seq import simple_seq2seq,seq2seq
from sklearn.model_selection import train_test_split
from data_generator import gen_add_data,add_data_decoder

def train_simple_seq2seq_add(x_train,y_train,x_test,y_test,encoder_dim=64,decoder_dim=32):
    
    print('在加法数据上测试简单seq2seq模型')
    
    in_len = x_train.shape[1]
    out_len = y_train.shape[1]
    feature_dim = x_train.shape[2]
    
    model = simple_seq2seq(in_len=in_len,out_len=out_len,feature_dim=feature_dim,
                           encoder_dim=encoder_dim,decoder_dim=decoder_dim)
    model.fit(x_train,y_train,epochs=10,batch_size=128,validation_split = 0.033)
    
    y_pred = model.predict(x_test)
    add_data_decoder(x_test,y_test,y_pred)
     
def train_seq2seq_add(x_train,y_train,x_test,y_test,encoder_dim=32,decoder_dim=32):
    
    print('在加法数据上测试标准seq2seq模型')
    
    in_len = x_train.shape[1]
    out_len = y_train.shape[1]
    feature_dim = x_train.shape[2]
    
    x_train1 = np.roll(y_train,1,axis=0)
    x_train1[0,:] = 0
    model = seq2seq(in_len=in_len,out_len=out_len,feature_dim=feature_dim,
                           encoder_dim=encoder_dim,decoder_dim=decoder_dim)
    model.fit([x_train,x_train1],y_train,epochs=10,batch_size=128,validation_split = 0.033)
    
    x_test1 = np.roll(y_test,1,axis=0)
    x_test1[0,:] = 0
    y_pred = model.predict([x_test,x_test1])
    add_data_decoder(x_test,y_test,y_pred)

def main(encoder_dim=16,decoder_dim=16):
    X,Y = gen_add_data(data_num=100000,max_single_num=200)
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.001)
    train_simple_seq2seq_add(x_train,y_train,x_test,y_test,encoder_dim=encoder_dim,
                             decoder_dim=decoder_dim)
    train_seq2seq_add(x_train,y_train,x_test,y_test,encoder_dim=encoder_dim,
                      decoder_dim=decoder_dim)
    
if __name__ == '__main__':
    main()
    