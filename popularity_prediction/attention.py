# -*- coding: utf-8 -*-
from keras import layers
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

def vector_mul(a,b):
    #a :(None,...,n)----batch_input like
    #b :(n,)---jist a vector
    #out :(None,...)
    return tf.reduce_sum(tf.multiply(a, b), reduction_indices=K.ndim(a)-1)

#test-vector_mul
'''
if __name__ == '__main__':
    x = tf.constant([[[1,2],[3,2]],[[4,2],[5,2]],[[1,3],[3,3]]])
    y = tf.constant([-1,1])
    mul = vector_mul(x,y)
    sess = tf.Session()
    print(sess.run(mul))
'''

#test-layers.dot
#situation:
#layers.dot([x,y],axes)
#x:batch_input like
#y:batch_input like
'''
if __name__ == '__main__':
    x = tf.constant([[[1,2],[3,2]],[[4,2],[5,2]],[[1,3],[3,3]]])
    y = tf.constant([[-1,1],[-2,1],[0,1]])
    mul = layers.dot([x,x],axes=1)
    sess = tf.Session()
    print(sess.run(mul))
'''
input_dim = 32

class self_attention_single(Layer):
    #attention = softmax(tanh(xW)v)
    #input shape:(None,time_step,feature_num)
    #x:(None,time_step,feature_num)
    #W:(feature_num,hidden_dim)
    #v:(hidden_dim,1)
    #attention:(None,time_step)
    #output:(None,feature_num)
    def __init__(self, hidden_dim, **kwargs):
        self.hidden_dim = hidden_dim
        super(self_attention_single, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        #print(input_shape)
        self.W = self.add_weight(name='W', 
                                      shape=(input_shape[-1], self.hidden_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.v = self.add_weight(name='v', 
                                      shape=(self.hidden_dim,),
                                      initializer='uniform',
                                      trainable=True)
        super(self_attention_single, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        att = K.softmax(vector_mul(K.tanh(K.dot(x, self.W)),self.v))
        #print(K.int_shape(att))
        x = K.permute_dimensions(x,(0,2,1))
        #print(K.int_shape(x))
        res = vector_mul(x,att)
        #print(K.int_shape(res))
        return res
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2])

def self_attention_multi(inputs):
    pass

class multiply_attention(Layer):
    #inputs:[H,s]
    #attention = softmax(HWs^T)
    #out = sum(attention_i*H_i)
    #H:(None,time_step,feature_num)
    #s:(None,current_dim)
    #W:(feature_num,current_dim)
    #output:(None,feature_num)
    
    def __init__(self, **kwargs):
        super(multiply_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W', 
                                      shape=(input_shape[0][2], input_shape[1][1]),
                                      initializer='uniform',
                                      trainable=True)
        super(multiply_attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        H,s = inputs
        att = layers.dot([K.dot(H, self.W),s],axes=-1)
        att = K.softmax(att)
        print(K.int_shape(att))
        res = layers.dot([H,att],axes=1)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])

def add_attention(H,s):
    pass

if __name__ == '__main__':
    from keras.layers import Input,Dense
    from keras.models import Model
    x = Input(shape=(10,12))
    z = Input(shape=(4,))
    y = Dense(11)(x)
    model = Model(inputs=x, outputs=y)
    model.summary()
    x_att = self_attention_single(11)(x)
    model = Model(inputs=x, outputs=x_att)
    model.summary()
    x_att1 = multiply_attention()([x,z])
    model = Model(inputs=[x,z], outputs=x_att1)
    model.summary()
