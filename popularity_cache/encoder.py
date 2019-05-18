# -*- coding: utf-8 -*-
from keras import layers
import keras.backend as K

def BiLSTM(input_shape,
           lstm_dim,
           return_sequences=False,
           activation='relu',
           dropout=0.2,
           recurrent_dropout=0.2,
           **kwargs):
    
    return layers.Bidirectional(layers.LSTM(lstm_dim, input_shape=input_shape,return_sequences=return_sequences,\
            activation=activation,dropout=dropout, recurrent_dropout=recurrent_dropout))

#默认padding补全
def multi_CNN(filters,
            kernel_size,
            dilation_rate=1,
            dropout=0.5,
            CNN_nums=3,
            residual=True,
            pooling=False,
            **kwargs):
    
    assert pooling in [False,'max','mean'],'pooling method invalid!'
    
    CNN_layers = [layers.Conv1D(filters=filters,
                                padding='same',
                                kernel_size=kernel_size,
                                data_format='channels_last',
                                dilation_rate=dilation_rate) for i in range(CNN_nums)]
    
    #在residual模式下保证维度一致
    helper_cnn = layers.Conv1D(filters=filters,
                               padding='same',
                               kernel_size=kernel_size,
                               data_format='channels_last',
                               dilation_rate=dilation_rate)
    
    def _multi_CNN(x):
        #记录所有CNN层的输出
        all_layers = []
        for one_layer in CNN_layers:
            x1 = one_layer(x)
            if residual:
                if K.int_shape(x) != K.int_shape(x1):
                    x = helper_cnn(x)
                x = x + x1
            else:
                x = x1
            x = layers.Dropout(dropout)(x)
            all_layers.append(x)
        if pooling == 'max':
            x = layers.GlobalMaxPooling1D()(x)
        return x
    
    return _multi_CNN

def Transformer():
    pass

if __name__ == '__main__':
    import numpy as np
    encoder = multi_CNN(filters=5,kernel_size=3,dilation_rate=1,CNN_nums=3)
    #encoder = BiLSTM(input_shape=(7,1),lstm_dim=16)
    x0 = K.variable(np.zeros((1,7,2))+2)
    x1 = K.variable(np.zeros((1,7,2))+2)
    x = x0 + x1
    y = encoder(x0)
    y1 = encoder(x0)
    print(K.eval(x0))
    print(K.eval(x))
    print(K.eval(y))
    print(K.eval(y1))
    print(K.int_shape(y))
