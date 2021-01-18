from functools import wraps

from keras.layers import (Add, Concatenate, Conv2D, MaxPooling2D, UpSampling2D,
                          ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from codes.core.utils import compose

#--------------------------------------------------#
#   单次卷积DarknetConv2D
#   正则化系数为5e-4
#   如果步长为2则自己设定padding方式。
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   DarknetConv2D + BatchNormalization + LeakyReLU
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------------------------#
#   残差结构
#   首先利用ZeroPadding2D和一个步长为2x2的卷积块进行高和宽的压缩
#   然后对num_blocks进行循环，循环内部是残差结构
def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)
        x = Add()([x,y])
    return x

#---------------------------------------------------------------------#
#   darknet53主体部分
#   输入为一张512x512x3的图片
#   输出为三个有效特征层
def darknet_body(x):
    # 512,512,3 -> 512,512,32
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    # 512,512,32 -> 256,256,64
    x = resblock_body(x, 64, 1)
    # 256,256,64 -> 128,128,128
    x = resblock_body(x, 128, 2)
    # 128,128,128 -> 64,64,256
    x = resblock_body(x, 256, 8)
    feat1 = x
    # 64,64,256 -> 32,32,512
    x = resblock_body(x, 512, 8)
    feat2 = x
    # 32,32,512 -> 16,16,1024
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1,feat2,feat3

