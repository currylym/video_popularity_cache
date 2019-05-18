# -*- coding: utf-8 -*-

'''
concat layer
'''
import keras.backend as K
from keras import layers

def simple_concat():
    return layers.Concatenate(axis=-1)

def gate_concat():
    def _concat(x):
        dim = K.int_shape(x)[-1]
        dense = layers.Dense(dim)
        weight = dense(x)
        x_gate = layers.Multiply()([x,weight])
        return x_gate
    return _concat
        
if __name__ == '__main__':
    import numpy as np
    simple_concat()
    gate_layer = gate_concat()
    x = K.variable(np.zeros((7,3))+2)
    y = gate_layer(x)
    print(K.eval(x))
    print(K.eval(y))