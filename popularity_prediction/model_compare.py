# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

import argparse
parser = argparse.ArgumentParser(description="")
parser.add_argument('--save_flag',default='',help='file save flag')
parser.add_argument('--repeat_time',default=1,type=int,help='model trained times')
parser.add_argument('--train_val_split_seed',default=1000,type=int,help='random seed to split train and val')
args = parser.parse_args()

#------------------------------prepare data------------------------------------
from utils import prepare_data_for_pp,save_result
x_train,y_train,x_test,y_test,x_val,y_val = prepare_data_for_pp(val_rate=0.1,seed=args.train_val_split_seed)

#-----------------------------simple baseline----------------------------------
from utils import plot_MSE_MAP_table
#mlr
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(x_train['main_data'][:,:,0],y_train)
y_pred_mlr = model.predict(x_test['main_data'][:,:,0])

#svr
from sklearn.svm import SVR
model = SVR()
model.fit(x_train['main_data'][:,:,0],y_train)
y_pred_svr = model.predict(x_test['main_data'][:,:,0])

#y=sum(x)
y_historyBase = x_test['main_data'][:,:,0].mean(axis = 1)

results = {
        'historyBased':y_historyBase,
        'mlr':y_pred_mlr,
        'svr':y_pred_svr,
        'y_test':y_test
        }

print('simple baseline finished')
plot_MSE_MAP_table(results,mode='mean first')

#assert 1==2
#-------------------------------build model------------------------------------
from numpy.random import seed
import matplotlib.pyplot as plt
from keras.callbacks import Callback,ReduceLROnPlateau,EarlyStopping

from model import build_model_att

def _mse(y_pred,y_test):
    return np.mean((y_pred-y_test)**2)

#该函数主要是为了画出在模型训练过程中train loss，val loss和test loss的变化情况
def predict(epochs=20,batch_size=256,**kwargs):
    
    seed(1)
    mymodel = build_model_att(**kwargs)
    
    class TEST_MSE(Callback):
        def __init__(self):
            self.x_test = x_test
            self.y_test = y_test
    
        def on_train_begin(self, logs={}):
            self.mse = []

        def on_epoch_end(self, batch, logs={}):
            def _mse(y_pred,y_test):
                return np.mean((y_pred-y_test)**2)
            epoch_end_predict = mymodel.predict(self.x_test)
            self.mse.append(_mse(epoch_end_predict[:,0],self.y_test))
    
    #回调函数
    mse_hist = TEST_MSE()
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=1e-5)
    #restore_best_weights=True--返回监控指标最好的模型
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2, 
                               restore_best_weights=True)
    
    history = mymodel.fit(x_train, y_train,epochs=epochs, batch_size=batch_size,
                          validation_data = (x_val,y_val),verbose=True,
                          callbacks=[mse_hist,reduce_lr,early_stopping])
    y_pred = mymodel.predict(x_test)
    
    #计算模型在测试集上的mse
    print('mse:%.4f' % _mse(y_pred[:,0],y_test))
    
    #画出训练集和验证集上的loss变化情况
    plt.figure()
    plt.plot(history.history['loss'],label='train_loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.plot(mse_hist.mse,label = 'test_loss')
    
    #分别画出val loss和test loss的最小值点
    def _highlight(x,y,color,loc,upx=0.8,upy=1.4,downx=0.8,downy=0.6):
        # 显示坐标点
        plt.scatter(x,y,s=20,marker='x')
        # 显示坐标点横线、竖线
        plt.vlines(x, 0, y, colors=color, linestyles="dashed", linewidth=1)
        plt.hlines(y, 0, x, colors=color, linestyles="dashed", linewidth=1)
        # 显示坐标点坐标值
        if loc == 'up':
            plt.annotate('(%d,%.4f)'%(x,y), xy=(x,y), xytext=(x*upx,y*upy), 
                         arrowprops=dict(facecolor='black', shrink=0.05, width=0.2, headwidth=0.5, headlength=0.5))
        if loc == 'down':
            plt.annotate('(%d,%.4f)'%(x,y), xy=(x,y), xytext=(x*downx,y*downy), 
                         arrowprops=dict(facecolor='black', shrink=0.05, width=0.2, headwidth=0.5, headlength=0.5))
        
    val_min_x = np.argmin(history.history['val_loss'])
    val_min_y = history.history['val_loss'][val_min_x]
    _highlight(val_min_x,val_min_y,color='black',loc='down')
    _highlight(val_min_x,mse_hist.mse[val_min_x],color='black',loc='up')
    
    test_min_x = np.argmin(mse_hist.mse)
    test_min_y = mse_hist.mse[test_min_x]
    _highlight(test_min_x,test_min_y,color='black',loc='up',upy=1.2)
    
    plt.grid()
    plt.legend()
    name = ['%s_%s' % (i.replace('_','-'),j) for i,j in kwargs.items()]
    name = '_'.join(name[:(len(name)//3)]) + '\n' + '_'.join(name[(len(name)//3):])
    name = name.replace('.','')
    plt.title(name)
    plt.savefig('result/fig/'+name,dpi=800)
    
    return y_pred

#-------------------------------default_param----------------------------------

# 在默认参数下跑不同特征时的模型：feature_chosen=0/1/2/3,use_iqiyi=True/False
# 默认参数
default_params = dict(epochs=100,
                      batch_size=128,
                      use_BN=False,
                      share_weight=True,
                      optimizer='adam',
                      lstm_activation='tanh',
                      dense_activation='selu',
                      lr=1e-2,
                      lstm_dim=8,
                      wide_mode=True)
# 在默认参数下重复
def default_param(repeat_time=1):
    results = {}
    for feature_chosen in [0,1,2,3]:
        for use_iqiyi in [True,False]:
            results['mse-%s-%d' % (use_iqiyi,feature_chosen)] = []
            for i in range(repeat_time):
                print('mse-%s-%d' % (use_iqiyi,feature_chosen),i)
                y_pred = predict(feature_chosen=feature_chosen,use_iqiyi=use_iqiyi,**default_params)
                results['mse-%s-%d' % (use_iqiyi,feature_chosen)].append(y_pred[:,0])
    return results

# 更新总的结果列表
deepmodel_default_param_result = default_param(repeat_time=args.repeat_time)
results.update(deepmodel_default_param_result)
tb1 = plot_MSE_MAP_table(results,mode='mean first')
tb2 = plot_MSE_MAP_table(results,mode='mean last')

# 保存评测结果
from parameters import TRAIN_START_DATE,TRAIN_END_DATE,TEST_START_DATE,TEST_END_DATE,UPDATE_CYCLE,HISTORY_NUM
result_dir = 'result/HistoryNum=%s_UpdateCycle=%s/' % (HISTORY_NUM, UPDATE_CYCLE)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
filename = result_dir + '%s_%s_%s_%s_default_%s.txt' % (TRAIN_START_DATE,TRAIN_END_DATE,TEST_START_DATE,TEST_END_DATE,args.save_flag)
save_result(tb1,filename)

# 保存不同方法的预测结果
for key in results:
    if isinstance(results[key],list):
        results[key] = results[key][0] #多次预测结果中随机找一个保存
results = pd.DataFrame(results)
filename1 = result_dir + '%s_%s_%s_%s_default_%s.csv' % (TRAIN_START_DATE,TRAIN_END_DATE,TEST_START_DATE,TEST_END_DATE,args.save_flag)
results.to_csv(filename1)
print('save prediction result!')

assert 1==2
#----------------------------------grid search---------------------------------

from grid_search import main
deep_results = main() #deep model得到的预测结果
results.update(deep_results)
plot_MSE_MAP_table(results,mode='mean first')

#---------------------------------similar test---------------------------------
for v in v_timeseries.index[5:10]:
    if v in similar_video:
        print(v)
        print(similar_video[v])
        plt.figure()
        plt.plot(np.array(v_timeseries.loc[v]),linewidth=2)
        for j in similar_video[v]:
            plt.plot(iqiyi_timeseries.loc[j],'--',linewidth=1)
        plt.xticks(np.arange(0,83,10))

