from scipy.fftpack import dct
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Permute

"""
A function other_transform() is implemented in utils.py which takes the input_feature, performs other statistic measures
instead of the global average pool and returns the transformed input.
Measures are:
Standard Deviation- transform='std',
Singular Value Decomposition- transform='svd',
Trace- transform='trace',
Discrete Cosine Transform- transform='dct_dcval'

"""

def other_transform(x_tensor, channel_axis, transform='std'):
       
    if(channel_axis == -1):
        x_tensor = Permute((3, 1, 2))(x_tensor) # Always Make channels first
    
    if transform=='std':
            # STANDARD DEVIATION
        std = tf.keras.backend.std(x_tensor, axis=[2,3]) # (batch, channels, widht, height)
        transformed_feature = std/tf.reduce_max(std, axis=None)
    
    elif transform=='dct_dcval':
        dct = tf.signal.dct(x_tensor, norm='ortho')
        transformed_feature = tf.keras.layers.GlobalMaxPool2D(data_format='channels_first')(dct) # dc term
        # DC term or the 1st term is always the largest in value, thus dct[0][0] === GlobalMaxpool2d

    elif transform=='svd':
        # SVD DECOMPOSITION
        s = tf.linalg.svd(x_tensor, compute_uv=False)
        transformed_feature = tf.reduce_max(s, axis=-1)
        transformed_feature = transformed_feature/tf.reduce_max(transformed_feature, axis=None) # Max value scaling.
 
    elif transform=='trace':
        # Sum of Diagonal
        trace = abs(tf.linalg.trace(x_tensor))
        transformed_feature = trace/tf.reduce_max(trace, axis=None) # Max value scaling.

    else:
        raise Exception("'{}' statistic measure is not implemented!".format(transform))

    return transformed_feature