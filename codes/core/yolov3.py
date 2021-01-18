from functools import wraps

from keras import backend as K
from keras.layers import (Add, Concatenate, Conv2D, MaxPooling2D, UpSampling2D,
                          ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from codes.core.darknet53 import darknet_body
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

# ---------------------------------------------------#
#   特征层->最后的输出
def make_last_layers(x, num_filters, out_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)

    # 将最后的通道数调整为outfilter
    y = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    y = DarknetConv2D(out_filters, (1, 1))(y)

    return x, y

# ---------------------------------------------------#
#   FPN网络的构建，并且获得预测结果
# ---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):
    # ---------------------------------------------------#
    #   生成darknet53的主干模型
    #   获得三个有效特征层，他们的shape分别是：
    #   64,64,256
    #   32,32,512
    #   16,16,1024
    # ---------------------------------------------------#
    feat1, feat2, feat3 = darknet_body(inputs)
    darknet = Model(inputs, feat3)
    # ---------------------------------------------------#
    #   第一个特征层
    #   y1=(batch_size,16,16,3,85)
    # ---------------------------------------------------#
    # 16,16,1024 -> 16,16,512 -> 16,16,1024 -> 16,16,512 -> 16,16,1024 -> 16,16,512
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    # 16,16,512 -> 16,16,256 -> 32,32,256
    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)

    # 32,32,256 + 32,32,512 -> 32,32,768
    x = Concatenate()([x, feat2])
    # ---------------------------------------------------#
    #   第二个特征层
    #   y2=(batch_size,32,32,3,85)
    # ---------------------------------------------------#
    # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    # 26,26,256 -> 26,26,128 -> 52,52,128
    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    # 52,52,128 + 52,52,256 -> 52,52,384
    x = Concatenate()([x, feat1])
    # ---------------------------------------------------#
    #   第三个特征层
    #   y3=(batch_size,52,52,3,85)
    # ---------------------------------------------------#
    # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])