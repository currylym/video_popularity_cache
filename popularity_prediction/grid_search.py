# -*- coding: utf-8 -*-
import time
import json
import numpy as np
from itertools import product
from keras.callbacks import ReduceLROnPlateau,EarlyStopping

from model import build_model_att
from utils import prepare_data_for_pp

def params_generator(params):
    keys = list(params.keys())
    values = list(params.values())
    all_param_set = product(*values)
    for param_set in all_param_set:
        yield dict(zip(keys,param_set))
        
def together_shuffle(X,y,seed=0):
    np.random.seed(seed)
    per = np.random.permutation(y.shape[0])
    X_shuffle = {}
    for key in X:
        X_shuffle[key] = X[key][per,:]
    y_shuffle = y[per,]
    return X_shuffle,y_shuffle

def _mse(y_pred,y_test):
    return np.mean((y_pred-y_test)**2)

def grid_search(x_train,y_train,x_val,y_val,use_feature_num=3,use_iqiyi=True,repeat_time=3):
    '''
    params:
    --------------------------------------------------------
      X_train
      y_train
      x_val
      y_val
      use_feature_num:
          0---only use history viewing data
          1---add tag info
          2---add tag/title info
          3---add tag/title/description info
      use_iqiyi:use iqiyi feature or not
      repeat_time:
      
    outputs:
    --------------------------------------------------------
      model:with best params
    '''
    np.random.seed(100)
    params_list=[
            dict(lstm_dim=[2,4,6,8],
                 use_BN=[False],
                 dense_dims=list(map(lambda x:[x]*use_feature_num,[2,4,6,8])),
                 activation=['relu'],
                 features=[['tags','title','description'][:use_feature_num]],
                 feature_shapes=[[300,300,300][:use_feature_num]],
                 optimizer=['adam'],
                 use_iqiyi=[use_iqiyi],
                 share_weight=[True,False]
                 ),
            dict(batch_size=[32],
                 epochs=[5]
                 )
            ]
    
    #param set1
    results = {}
    start = time.time()
    for param in params_generator(params_list[0]):
        print('  %s' % str(param))
        results[str(param)] = []
        np.random.seed(999)
        model = build_model_att(**param)
        for i in range(repeat_time):
            #打乱训练集
            x_shuffle,y_shuffle = together_shuffle(x_train,y_train,i*100)
            #定义回调函数
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, 
                                          min_lr=0.001)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2, 
                               restore_best_weights=True)
            #训练模型
            history = model.fit(x_shuffle, y_shuffle, epochs=20, batch_size=256,
                                validation_data=(x_val,y_val),verbose=0, shuffle=True,
                                callbacks=[reduce_lr,early_stopping])
            print('  repeat %d' % (i+1))
            print(history.history['val_loss'],'%.4f min' % ((time.time()-start)/60))
            #使用此时的模型对验证集进行预测并计算loss
            y_val_pred = model.predict(x_val)
            results[str(param)].append(_mse(y_val,y_val_pred))
    #计算模型loss的平均值和方差，打印平均值最低的模型参数     
    means = {key:np.mean(value) for key,value in results.items()}
    stds = {key:np.std(value) for key,value in results.items()}
    for param in means:
        print('param : %s | mean : %.4f | std : %.4f' % (param,means[param],stds[param]))
    
    best_param = eval(min(list(means.items()),key=lambda x:x[1])[0])
    print('best param : %s' % str(best_param))
    
    best_model = build_model_att(**best_param)
    best_model.fit(x_train,y_train,epochs=20, batch_size=256,
                                validation_split=0,verbose=0, shuffle=True,
                                callbacks=[reduce_lr])
    return best_model,results

def main(repeat_time=3):
    '''
    主程序：在是否使用iqiyi视频/文本特征使用个数（0/1/2/3）等情况下寻找最优的模型结构
    
    params:
    -------
    repeat_time:多次训练模型得到测试集的预测结果，防止模型训练不稳定
    '''
    x_train,y_train,x_test,y_test,x_val,y_val = prepare_data_for_pp()
    
    results = {}
    search_results = {}
    
    for use_iqiyi in [True,False]:
        for use_feature_num in range(4):
            best_model,_ = grid_search(x_train,y_train,x_val,y_val,
                                       use_feature_num=use_feature_num,
                                       use_iqiyi=use_iqiyi,repeat_time=3)
            y_pred = best_model.predict(x_test)
            results['mse-%s-%d' % (use_iqiyi,use_feature_num)] = y_pred[:,0]
            search_results['mse-%s-%d' % (use_iqiyi,use_feature_num)] = _
            
    with open('search_res.json','w') as writer:
        writer.write(json.dumps(search_results,indent=1,ensure_ascii=False))
            
    return results
    
    
    
    
            
    
    
    
    
