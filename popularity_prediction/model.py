# -*- coding: utf-8 -*-

from attention import self_attention_single,multiply_attention
from keras.layers import Input,LSTM,concatenate,Dense,Bidirectional,BatchNormalization,Conv1D,Activation,Dropout,Add
from keras.models import Model
from keras.layers.pooling import GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers.core import Reshape,Flatten
from keras.optimizers import Adam,SGD
from keras import backend as K
import tensorflow as tf

import sys
sys.path.append('..')
from  parameters import HISTORY_NUM
time_step = HISTORY_NUM

#attention based model
def build_model_att(
        attention_type='multiply',
        att_dim=10,
        loss_type='mse',
        lstm_dim=4,
        lstm_layers=1,
        dense_dims=[2,2,2],
        optimizer='adam',
        use_BN=False,
        features=['tags','title','description'],
        feature_shapes=[300,300,300],
        feature_chosen=3,
        use_iqiyi=True,
        lstm_activation='relu',
        dense_activation='relu',
        share_weight=True,
        lr=1e-3,
        dropout=0.5,
        sgd_momentum=0.0,
        sgd_decay=0.0,
        wide_mode=False,
        embedding_mode=False,
        use_CNN=False
        ):
    '''
    --------------------------------------------------
    important params:
        optimizer:优化器
        lr:学习率
        use_BN:是否使用batchnorm
        activation:激活函数
        share_weight:是否权重共享
        lstm_dim:lstm的隐藏层维数
        dense_dims:三组文本特征分别的使用的dense层维数
        wide_mode:是否结合浅层模型
        embedding_mode:是否对整数值的流行度进行embedding
        use_CNN:是否对输入的时间序列进行一维卷积(可以理解为滑动加权，降低序列的不平稳性)
    ---------------------------------------------------
    '''
    # 对特征进行截取
    features = features[:feature_chosen]
    feature_shapes = feature_shapes[:feature_chosen]
    dense_dims = dense_dims[:feature_chosen]
    
    # 根据是否共享参数初始化lstm层和dense层
    if share_weight:
        _encoder_layer = Bidirectional(LSTM(lstm_dim, input_shape=(time_step, 1),return_sequences=False,\
                              activation=lstm_activation,dropout=0.2, recurrent_dropout=0.2))
        encoder_layers = [_encoder_layer]*2
        if len(set(dense_dims)) != 1:
            dense_layers = [Dense(dim) for dim in dense_dims]
        else:
            _dense_layer = Dense(dense_dims[0])
            dense_layers = [_dense_layer]*len(dense_dims)
    else:
        encoder_layers = [Bidirectional(LSTM(lstm_dim, input_shape=(time_step, 1),return_sequences=False,\
                              activation=lstm_activation,dropout=0.2, recurrent_dropout=0.2)) for i in range(2 if use_iqiyi else 1)]
        dense_layers = [Dense(dim) for dim in dense_dims]    
    
    # 对主要的时间序列特征进行lstm->dropout->batchnorm
    main_input = Input(shape=(time_step,1), name='main_data')
    if use_CNN:
        cnn_out = Conv1D(filters=1,kernel_size=3,padding='same')(main_input)
    else:
        cnn_out = main_input
    lstm_out = encoder_layers[0](cnn_out)
    lstm_out = Dropout(dropout)(lstm_out)
    if use_BN : lstm_out = BatchNormalization()(lstm_out)

    inputs = [main_input]
    numerical_features = lstm_out
    # 处理iqiyi特征:attention->lstm->dropout->batchnorm
    if use_iqiyi:
        iqiyi_input = Input(shape=(10,time_step),name='similar_iqiyi_video_ts')
        inputs.append(iqiyi_input)
        if attention_type == 'multiply':
            att_iqiyi = multiply_attention()([iqiyi_input,lstm_out])
            att_iqiyi = Reshape((time_step,1))(att_iqiyi)
            #print(K.int_shape(att_iqiyi))
            att_iqiyi_lstm = encoder_layers[1](att_iqiyi)
            att_iqiyi_lstm = Dropout(dropout)(att_iqiyi_lstm)
            if use_BN : att_iqiyi_lstm = BatchNormalization()(att_iqiyi_lstm)
            numerical_features = concatenate([numerical_features,att_iqiyi_lstm])

    em_features = []
    # 处理文本特征:dense->activation->dropout->batchnorm
    for feature,shape,dense_layer in zip(features,feature_shapes,dense_layers):
        em_input = Input(shape=(shape,), name=feature)
        em_dense = dense_layer(em_input)
        em_dense = Activation(dense_activation)(em_dense)
        em_dense = Dropout(dropout)(em_dense)
        if use_BN : em_dense = BatchNormalization()(em_dense)
        inputs.append(em_input)
        em_features.append(em_dense)
    
    # 拼接所有特征
    if len(em_features) > 1:
        em_features = concatenate(em_features)
        x = concatenate([numerical_features,em_features],name='total_features')
    elif len(em_features) == 1:
        x = concatenate([numerical_features,em_features[0]],name='total_features')
    else:
        x = numerical_features
    #x = Dropout(0.1)(x)

    # 最后添加dense层
    main_output = Dense(1, activation='softplus', name='main_output')(x) # 非负预测
    
    # 融合低阶模型
    if wide_mode:
        main_input1 = Flatten()(cnn_out)
        wide_output = Dense(1, activation='softplus', name='wide_output')(main_input1)
        main_output = Add()([main_output,wide_output])

    model = Model(inputs=inputs,outputs=main_output)

    # 定义优化器
    if optimizer == 'adam':
        model.compile(optimizer=Adam(lr=lr), loss=loss_type)
    if optimizer == 'sgd':
        model.compile(optimizer=SGD(lr=lr, momentum=sgd_momentum, decay=sgd_decay), loss=loss_type)
    model.summary()
    return model

if __name__ == '__main__':
    #build_model(use_iqiyi=True)
    build_model_att(use_iqiyi=True,
                    share_weight=True,
                    feature_chosen=1,
                    wide_mode=True,
                    use_CNN=True)
