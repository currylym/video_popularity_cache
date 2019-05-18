# -*- coding: utf-8 -*-
'''
keras版本
'''
import keras
import keras.backend as K 

def BiLSTM(input_shape,
           lstm_dim,
           return_sequences=False,
           activation='relu',
           dropout=0.2,
           recurrent_dropout=0.2,
           **kwargs):
    
    return keras.layers.Bidirectional(layers.LSTM(lstm_dim, input_shape=input_shape,return_sequences=return_sequences,\
            activation=activation,dropout=dropout, recurrent_dropout=recurrent_dropout))

#默认padding补全
def multi_CNN(filters=3,
            kernel_size=3,
            dilation_rate=1,
            dropout=0.5,
            CNN_nums=3,
            residual=True,
            pooling=False,
            **kwargs):
    
    assert pooling in [False,'max','mean'],'pooling method invalid!'
    
    CNN_layers = [keras.layers.Conv1D(filters=filters,
                                padding='same',
                                kernel_size=kernel_size,
                                data_format='channels_last',
                                dilation_rate=dilation_rate) for i in range(CNN_nums)]
    
    #在residual模式下保证维度一致
    helper_cnn = keras.layers.Conv1D(filters=filters,
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
            x = keras.layers.Dropout(dropout)(x)
            all_layers.append(x)
        if pooling == 'max':
            x = keras.layers.GlobalMaxPooling1D()(x)
        return x
    
    return _multi_CNN

def TCNN(**kwargs):
    pass

if __name__ == '__main__':
    x = keras.layers.Input(shape=(7,3))
    encoder = multi_CNN(pooling='max')
    y = keras.layers.Lambda(encoder)(x)
    y = keras.layers.Dense(1)(y)
    print(K.int_shape(x))
    print(K.int_shape(y))
    model = keras.models.Model(inputs=x,outputs=y)
    model.summary()
