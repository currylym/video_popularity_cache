# -*- coding: utf-8 -*-

from encoder import BiLSTM,multi_CNN,Transformer
from attention import multiply_attention,add_attention
from concat import simple_concat as gate_concat

import keras.backend as K
from keras.layers import Input,Dense,Lambda,Reshape
from keras.models import Model
from keras.optimizers import Adam

def pairwise_loss(_, y_pred):
    margin = K.constant(0.5)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,1,0]) - K.square(y_pred[:,0,0]) + margin))

def accuracy(_, y_pred):
    return K.mean(y_pred[:,0,0] > y_pred[:,1,0])

def RankModel(
        features_name = ['title','tags','description','history_viewing','related_series'],
        features_shape = [(8,500),(4,500),(30,500),(7,1),(10,7)],
        seq_encoder = 'BiLSTM',
        related_series_att = 'multiply_attention',
        text_encoder = 'multi_CNN',
        is_pairwise = True,
        loss_type = 'mse',
        **kwargs
        ):
    
    #参数有效性检查
    assert seq_encoder in ['BiLSTM','multi_CNN','Transformer'],'encoder invalid'
    assert text_encoder in ['BiLSTM','multi_CNN','Transformer'],'encoder invalid'
    
    def TextEncoder():
        if text_encoder == 'BiLSTM':
            encoder = BiLSTM(**kwargs)
        elif text_encoder == 'multi_CNN':
            encoder = multi_CNN(**kwargs)
        elif text_encoder == 'Transformer':
            encoder = Transformer(**kwargs)
        else:
            print("please choose text_encoder in ['BiLSTM','multi_CNN','Transformer']")
        return encoder
    
    def SequenceEncoder():
        if seq_encoder == 'BiLSTM':
            encoder = BiLSTM(**kwargs)
        elif seq_encoder == 'multi_CNN':
            encoder = multi_CNN(**kwargs)
        else:
            print("please choose seq_encoder in ['BiLSTM','multi_CNN']")
        return encoder
    
    def Att():    
        if related_series_att == 'multiply_attention':
            return multiply_attention()
        if related_series_att == 'add_attention':
            return add_attention()
    
    #模块初始化，共享参数
    text_en = TextEncoder() #文本编码器
    sequence_en = SequenceEncoder() #序列编码器
    apply_attention = Att() #attention函数
    concat = gate_concat() #concat函数
    dense = Dense(1) #dense层
    
    def _encoder_concat_dense(inputs):
        text_info = []
        for i in range(3):
            text_info.append(text_en(inputs[i]))
                
        text_info.append(sequence_en(inputs[3]))
            
        if related_series_att == 'multiply_attention':
            print(K.int_shape(inputs[4]))
            print(K.int_shape(text_info[3]))
            att_related_series = apply_attention([inputs[4],text_info[3]])
            att_related_series = Reshape((-1,1))(att_related_series)
            print(K.int_shape(att_related_series))
            text_info.append(sequence_en(att_related_series)) 
        out = dense(concat(text_info))
        return out
        
    if is_pairwise:
        
        positive_inputs = [Input(shape=fs,name='pos_'+fn) for fn,fs in zip(features_name,features_shape)]
        negative_inputs = [Input(shape=fs,name='neg_'+fn) for fn,fs in zip(features_name,features_shape)]
        
        pos_score = _encoder_concat_dense(positive_inputs)
        neg_score = _encoder_concat_dense(negative_inputs)
        
        def cal_output_shape(input_shape):
            shape = list(input_shape[0])
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)

        stacked_dists = Lambda(
            lambda vects: K.stack(vects, axis=1),
            name='stacked_dists',
            output_shape=cal_output_shape
            )([pos_score, neg_score])

        model = Model(positive_inputs+negative_inputs, stacked_dists, name='pairwise model')
        model.compile(loss=pairwise_loss, optimizer=Adam(lr=0.01), metrics=[accuracy])
        
        return model
    
    else:
        
        inputs = [Input(shape=fs,name=fn) for fn,fs in zip(features_name,features_shape)]
        score = _encoder_concat_dense(inputs)
        
        model = Model(inputs, score, name='pointwise model')
        model.compile(loss='mse', optimizer=Adam(lr=0.01))
        
        return model

if __name__ == '__main__':
    mymodel = RankModel(
            features_shape=[(300,1),(300,1),(300,1),(7,1),(10,7)],
            input_shape=(7,1),
            lstm_dim=16,
            filters=8,
            pooling='max',
            kernel_size=3,
            residual=False,
            dilation_rate=2)
    mymodel.summary()
