# -*- coding: utf-8 -*-

from keras.models import Sequential,Model
from keras.layers import LSTM,RepeatVector,TimeDistributed,Dense,Input,Concatenate

def simple_seq2seq(in_len,out_len,feature_dim,encoder_dim=64,decoder_dim=64):
    '''
    simple seq2seq:解码器每步的输入为编码器最后的输出
    
    params:
    -------
    encoder_dim:编码器的隐藏层维数
    decoder_dim:解码器的隐藏层维数
    in_len:输入序列的长度
    out_len:输出序列的长度
    feature_dim:每个时间步的特征维数
    
    '''
    model = Sequential() 
    model.add(LSTM(encoder_dim, input_shape=(in_len, feature_dim)))
    model.add(RepeatVector(out_len))
    model.add(LSTM(decoder_dim, return_sequences=True))
    model.add(TimeDistributed(Dense(feature_dim, activation= 'softmax' ))) 
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    print(model.summary())
    return model

def seq2seq(in_len,out_len,feature_dim,encoder_dim=64,decoder_dim=64,peek=True):
    '''
    标准seq2seq:解码器初始状态为编码器的最后状态，初始输入为编码器最后的输出/全零向量，之后将前一时间步的
               准确结果作为下一时间步的输入。
               <!>问题：在训练的时候可以将前一时间步的准确结果作为下一时间步的输入，而在预测时则只能将前一时间步
                        的预测结果作为下一时间步的输入。
               
    params:
    -------
    encoder_dim:编码器的隐藏层维数
    decoder_dim:解码器的隐藏层维数
    in_len:输入序列的长度
    out_len:输出序列的长度
    feature_dim:每个时间步的特征维数
    peek:是否在解码器每步的输入加上编码器最后的输出
    
    注意：编码器隐藏层维数要和解码器一致
    '''
    assert encoder_dim == decoder_dim,'编码器隐藏层维数和解码器不一致'
    encoder_inputs = Input(shape=(in_len, feature_dim))
    encoder = LSTM(encoder_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(out_len, feature_dim))
    if peek:
        encoder_outputs_repeat = RepeatVector(out_len)(encoder_outputs)
        decoder_inputs_concat = Concatenate()([decoder_inputs,encoder_outputs_repeat])
    else:
        decoder_inputs_concat = decoder_inputs
    decoder_lstm = LSTM(decoder_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs_concat,
                                     initial_state=encoder_states)
    decoder_dense = Dense(feature_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    print(model.summary())
    return model

if __name__ == '__main__':
    model = simple_seq2seq(encoder_dim=64,decoder_dim=64,in_len=10,out_len=10,feature_dim=10)
    model = seq2seq(encoder_dim=64,decoder_dim=64,in_len=10,out_len=10,feature_dim=10)