# This is the version used by version v4_6
# Dec 29, 2020
# model builder
# import model_build


# refer to
# https://machinelearningmastery.com/tensorflow-tutorial-deep-learning-with-tf-keras/
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough
# https://www.tensorflow.org/tutorials/quickstart/advanced

import math
import numpy as np
import logging

import tensorflow as tf

from keras import Sequential,layers
from keras.layers import Dense
from keras.layers import BatchNormalization

from keras.layers import Dropout
from keras.layers import LayerNormalization
from keras.layers import MultiHeadAttention
from keras import Input,Model
from keras.layers import Layer, Conv2D
from keras.layers import Embedding
from keras import backend as K

## **************************************
def embedding_f(inputx,layer_n=3, unit=128):
    # He 法线初始化器用于初始化全连接层的内核权重。
    he = tf.keras.initializers.HeNormal()
    # 变量 x 最初设置为 inputx。
    x = inputx
    # 循环执行layer_n次-对应于定义的层数。
    # 在每次迭代中，都会向模型添加一个具有单位数量的单元和 ReLU 激活函数的全连接层。 该层的内核权重使用 He 法线初始化器进行初始化。
    # 全连接层的输出被分配回变量x。
    for i in range(layer_n):
        # x = Dense(units=unit//2**(layer_n-i-1),activation="relu", kernel_initializer=he)(x)
        x = Dense(units=unit,activation="relu", kernel_initializer=he)(x)
        # 在每个全连接层之后，添加批量归一化层以对激活进行归一化--稳定训练过程并提高模型的性能。
        x = BatchNormalization() (x)
    # 该函数返回最终的输出张量 x，它表示输入张量经过指定数量的全连接层和批量归一化层后的嵌入。
    return x

## **************************************
def decoder_f(inputx,layer_n=3, unit=128):
    he = tf.keras.initializers.HeNormal()
    x = inputx
    for i in range(layer_n):
        # x = Dense(units=unit//2**i,activation="relu", kernel_initializer=he)(x)
        x = Dense(units=unit,activation="relu", kernel_initializer=he)(x)
        x = BatchNormalization() (x)
    return x

# check test_mask.py to see why this is like adding two new axises
def create_padding_mask_any(inputs0, mask_value=0):
    print(inputs0.shape)
    # 首先，该函数使用 np.isnan(mask_value) 检查 mask_value 是否为 NaN。 如果 mask_value 为 NaN，则该函数应用以下过程：
    # seq = tf.cast(tf.math.not_equal(inputs[:, :, 0], mask_value), tf.float32)
    if np.isnan(mask_value):
        # 使用 tf.math.logic_not(tf.math.is_nan(inputs0)) 检查 input0 张量中的任何元素是否不是 NaN。
        # 然后通过应用逻辑或运算 (tf.math.reduce_any(...)) 来减少沿张量第三个轴 (axis=2) 的值。
        # 使用 tf.cast(..., tf.float32) 将生成的张量转换为 float32，并分配给变量 seq。
        seq = tf.cast(tf.math.reduce_any(tf.math.logical_not(tf.math.is_nan(inputs0)),axis=2), tf.float32)
    # 如果 mask_value 不为 NaN，则应用以下过程：
    else:
        # 使用 tf.math.not_equal(inputs0, mask_value) 检查 input0 张量中的每个元素是否不等于 mask_value。
        # 然后通过应用逻辑或运算 (tf.math.reduce_any(...)) 来减少沿张量第三个轴 (axis=2) 的值。
        # 使用 tf.cast(..., tf.float32) 将生成的张量转换为 float32，并分配给变量 seq。
        # seq = tf.cast(tf.math.reduce_any(tf.math.not_equal(inputs0, mask_value),axis=2), tf.float32)
        seq = tf.cast(tf.math.reduce_any(tf.math.not_equal(inputs0, mask_value), axis=(2, 3, 4)), tf.float32)
        # inputs_tf = tf.constant(inputs0)
        # mask_value = -9999
        # # 创建一个与input_images_train_norm2维度相同的掩码张量
        # mask = tf.constant(mask_value, shape=inputs_tf[:, :, :, 0:50, 0:50].shape, dtype=inputs_tf.dtype)
        # # 将掩码张量转换为与inputs_tf相同的数据类型
        # mask = tf.cast(mask, dtype = inputs_tf.dtype)
        # # 判断是否所有元素都等于掩码值
        # # value_mask = tf.reduce_all(tf.equal(inputs_tf[:, :, :, 0:50, 0:50], mask), axis=(2, 3, 4))
        # value_mask = tf.logical_not(tf.reduce_all(tf.equal(inputs_tf[:, :, :, 0:50, 0:50], mask), axis=(2, 3, 4)))
        # # 转换为float32类型
        # seq = tf.cast(value_mask, tf.float32)
        # # Convert 0s to -9999 and leave 1s as 1
        # seq = tf.where(tf.equal(seq, 0), tf.constant(-9999, dtype=tf.float32), seq)
    # 返回两个张量：
    # padding_mask 的初始形状为 [17, 80, 1]，通过使用 [:, :, tf.newaxis, tf.newaxis, :]，在第三个和第四个位置插入了新的维度。
    # 结果形状为 [17, 80, 1, 1, 1]，其中前两个维度与初始形状保持不变，而新插入的维度分别为 1，1 和 1。这意味着在这些新插入的维度上，原先的值在复制的过程中会延展和重复。
    # 这种操作常用于在张量的不同位置扩展维度，以便在进行一些操作（例如广播）时能够匹配形状
    # seq[:, :,tf.newaxis, tf.newaxis, :]：该张量的形状为 (batch_size, 1, 1, seq_len)，表示 3D 操作的填充掩码。
    # seq[:, :, tf.newaxis]：该张量的形状为 (batch_size, seq_len, 1)，表示 2D 操作的填充掩码。
    return seq[:,tf.newaxis,tf.newaxis,:],seq[:,:,tf.newaxis]  # (batch_size, timeseries, seq_len)

class CustomLayer(Layer):
    def __init__(self):
        super(CustomLayer, self).__init__()
    
    def call(self, inputs):
        mask_value = -9999
        mask = tf.constant(mask_value, shape=inputs.shape, dtype=inputs.dtype)
        mask = tf.cast(mask, dtype=inputs.dtype)
        value_mask = tf.logical_not(tf.reduce_all(tf.equal(inputs, mask), axis=(2, 3, 4)))
        seq = tf.cast(value_mask, tf.float32)
        
        return seq[:, tf.newaxis, tf.newaxis, :], seq[:, :, tf.newaxis]

# 函数positional_encoding 将长度和深度作为输入。 它返回一个位置编码张量。
def positional_encoding(length, depth):
    # 变量深度更新为等于深度整除 2。
    depth = depth//2
    # 使用 np.arange(length)[:, np.newaxis] 创建变量位置。 它是一个形状为 (length, 1) 的 2D numpy 数组，并以柱状格式包含从 0 到 length-1 的数字序列。
    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    # 使用 np.arange(深度)[np.newaxis, :]/深度 创建变量深度。 它是一个形状为 (1, 深度) 的 2D numpy 数组，包含从 0 到深度-1 除以深度的值。
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
    # angle_rates 变量的计算公式为 1 / (10000**深度)。 它是一个形状为 (1, 深度) 的 2D numpy 数组，表示每个深度值的角速率。
    angle_rates = 1 / (10000**depths)         # (1, depth)
    # angle_rads 变量的计算方式为：positions * angle_rates。 它是一个具有形状（长度、深度）的 2D numpy 数组，表示每个位置和深度组合的角度（以弧度为单位）。
    angle_rads = positions * angle_rates      # (pos, depth)
    # 使用 np.concatenate 连接沿最后一个轴的angle_rads 的正弦和余弦函数来创建pos_encoding 变量。 生成的张量具有形状（长度、深度*2）。
    pos_encoding = np.concatenate( [np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    # 使用 tf.cast 将 pos_encoding 张量转换为 tf.float32 并作为位置编码张量返回
    return tf.cast(pos_encoding, dtype=tf.float32)

# 函数 point_wise_feed_forward_network 使用两个全连接层创建逐点前馈网络

##************************************************************************************************************
## point_wise_feed_forward_network
def point_wise_feed_forward_network(d_model, dff, reg=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', kernel_regularizer=reg),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model, kernel_regularizer=reg)  # (batch_size, seq_len, d_model)
    ])

#def point_wise_feed_forward_network(d_model, dff):
    # 它有两个参数 d_model 和 dff ，分别表示输入维度和中间层的维度
    # 在函数内部，创建了一个 tf.keras.Sequential 模型，它是一个线性的层堆栈。
    # 该模型由两个全连接层组成。 第一个全连接层具有 dff 个单元并使用 ReLU 激活函数。 该层的输出形状为(batch_size, seq_len, dff)。
    # 第二个全连接层有 d_model 个单元。 该层的输出形状为(batch_size, seq_len, d_model)。
    # 最后返回创建的表示逐点前馈网络的顺序模型。
    # return tf.keras.Sequential([
        # tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        # tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    # ])

## check test_mask.py to see why this is like adding two new axises
def create_padding_mask(inputs,mask_value=0):
    # 通过检查输入张量的第一个通道 (inputs[:,:,0]) 中的每个元素是否不等于 mask_value 来创建填充掩码。
    #  not_equal 函数将每个元素与 mask_value 进行比较，并返回与输入[:,:,0] 形状相同的布尔张量。
    #  然后使用cast函数将布尔值转换为float32值，其中1表示该元素不等于mask_value，0表示该元素等于mask_value。
    # seq = tf.cast(tf.math.not_equal(inputs, mask_value), tf.float32)
    seq = tf.cast(tf.math.reduce_any(tf.math.not_equal(inputs, mask_value), axis=2), tf.float32)
    new_seq = tf.where(tf.equal(seq, 0), tf.constant(-1e9, dtype=tf.float32), tf.cast(seq, tf.float32))
    # 在这里应该注意将被填充的位置设置为-9999，还是和最开始CNN的mask一样，但是需要注意的是经过CNN的操作，x中被填充的地方变成了0
    # 如果要进行mask 需要把输入变量x1 中为0的地方将padding_mask标记为-9999
    # 生成的张量 seq 的形状为 (batch_size, seq_len) 并表示填充掩码。
    # 最后，该函数返回填充掩码张量，并使用 [:, tf.newaxis, tf.newaxis, :] 添加两个附加维度。
    # 生成的张量的形状为 (batch_size, 1, 1, seq_len)。 此格式通常用于在 3D 操作中应用填充掩码。
    return new_seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

## *******************
## embedding inputs 
def encoder_embed (units, encoder_layer=1, embed_n=0):
    if encoder_layer==1:
        return Dense(units, use_bias=False) if embed_n==0 else Embedding(embed_n, units)
    else:
        list_layers = []
        for ill in range(encoder_layer):
            if ill==0 and embed_n>0:
                list_layers.append (Embedding(embed_n, units)) 
                list_layers.append (tf.keras.layers.ReLU()) 
            else:
                list_layers.append (Dense(units, activation='relu')) 
        
        # list_layers.append (Dense(units)) 
        
        return tf.keras.Sequential(list_layers) 


# layer_cnn=2; layern=3; units=32; n_times=input_images_train_norm3.shape[1]; 
# n_times_out=1; n_window=3; n_head=4; drop=0.001; n_out=3; 
# n_features=15; is_batch=True; mask_value=-9999.0; 
# is_zero_fill=True; doy_encoder=1; 
# is_rptv = True; is_sensor = False; is_rptv_sensor=False
# 
# inputs = (input_images_train_norm3 [:2,:,:,:,:], input_covariate_train_norm3[:2,:])
# inputs = (input_images_train_norm3 [:1,:,:,:,:], input_covariate_train_norm3[:1,:])
# a = model.predict(inputs)
# inputs[0] [0,:20,1,1,1]
# x  [0,:20,1,1,1]
# x  [0,:20,0,0,1]
# x  [0,:20,1]
def get_transformer_transformer(layer_cnn=2, layern=3, units=64, n_times=80,
                        n_times_out=1, n_window=3,n_head=4, drop=0.1,n_out=1,
                        n_features=15, is_batch=True, mask_value=-9999.0,is_zero_fill=True, doy_encoder=0,
                        is_rptv = False,is_sensor =False):
    # 定义了一个Transformer-CNN模型，它是一个混合模型，同时接收图像数据和非图像数据作为输入，并将它们融合以生成最终的输出.这样设计的模型可以在处理复杂任务时融合多种数据源的信息，以提高模型的性能和泛化能力。
    # 定义了两个输入，一个是形状为(n_times, n_window, n_window, 7)的图像数据（inputs[0]）:
    # It appears that the model expects a 5-dimensional input where the first dimension represents the number of samples (None),
    # followed by three dimensions for the width, height, and channels (50, 50, 7),
    # and the last dimension represents another channel or feature (377).
    # 另一个是形状为(n_times, n_doy)的非图像数据（inputs[1]）。
    # inputs = [Input(shape=(n_times, n_window, n_window, 7)), Input(shape=(n_times, n_doy))]
    inputs = [Input(shape=(n_times,n_window,n_window,6)),Input(shape=(n_times,n_features))]
    # x = inputs[0]
    # 创建用于训练数据 填充掩码的padding_mask和padding_mask3d 在这里之前应该确保x1完全为-9999时，x2或者其他的也应该是-9999
    # 这里还需要用x变量作mask 判断依据就是如果input_images_train_norm3第i条记录 第j天 第m个波段的值全为-9999，
    # 那么就将这条记录masked
    # padding_mask, padding_mask3d = create_padding_mask_any(inputs[1], mask_value)
    padding_mask_cnn, padding_mask3d_cnn = create_padding_mask_any(inputs[0], mask_value)
    # 下面这是为什么要进行判断？判断之后第2个为啥要这样处理
    if not is_zero_fill:
        x = inputs[0]
    else:
        x = inputs[0] * padding_mask3d_cnn[:, :, tf.newaxis, tf.newaxis, :]
    
    ## *******************
    ## encoder 
    x = encoder_embed (units, encoder_layer=1)(x)
    x2 = tf.reshape(x        ,shape=(-1,n_times,n_window*n_window,units) )
    i2_formask = tf.reshape(inputs[0],shape=(-1,n_times,n_window*n_window,6    ) )
    ## *******************
    ## spatial position encoding 
    coords_h = np.arange(n_window); coords_w = np.arange(n_window)
    coords_matrix = np.meshgrid(coords_h, coords_w, indexing='ij');
    coords = np.stack(coords_matrix)
    coords = tf.cast(tf.transpose(coords, perm=(1,2,0)),tf.float32)
    coords = (coords-tf.math.reduce_mean(coords))/tf.math.reduce_std(coords)
    
    xposition = tf.reshape(encoder_embed (units, encoder_layer=1)(coords),[n_window*n_window,units]) 
    
    ## 处理非图像数据并进行嵌入(embedding)操作，将每个时间步的非图像数据转换为与图像数据具有相同维度的特征表示。
    ## *******************
    ## *******************
    # temporal positional 位置编码的mask
    mask_multi2 = tf.cast(tf.math.not_equal(inputs[1][:, :, 0:1], mask_value), tf.float32)
    doy_array = np.array(range(1,366))
    if not is_zero_fill:
        xpp = inputs[1][:, :, 0:1]
    else:
        xpp = inputs[1][:, :, 0:1]
        xpp = (xpp-doy_array.mean()) /doy_array.std()
        xpp = xpp * mask_multi2 * padding_mask3d_cnn
        xrptv = inputs[1][..., 1:5] * mask_multi2 * padding_mask3d_cnn
        xsensor = inputs[1][:, :, 5:6] * mask_multi2 * padding_mask3d_cnn
    
    embedding_p = encoder_embed (units, encoder_layer=1)    
    if is_rptv:
        xtp = embedding_p(tf.concat([xpp,xrptv],axis=2))
    else:
        xtp = embedding_p(xpp)

    embedding_sensor = Dense(units,encoder_layer=1)
    if is_sensor:
        x = x + embedding_sensor(xsensor)
    
    ## *******************
    ## define spatial transformer layers
    atten_layer = MultiHeadAttention(key_dim=units // n_head, num_heads=n_head)
    pwffn_layer = point_wise_feed_forward_network(units, units * 4)
    drop1 = Dropout(drop)
    drop2 = Dropout(drop)
    ln1   = LayerNormalization(epsilon=1e-6)
    ln2   = LayerNormalization(epsilon=1e-6)
    for ti in range(n_times):
        # x3 = x2[:,ti,:,:]+xposition[tf.newaxis,:,:]
        x3 = x2[:,ti,:,:]+xposition[tf.newaxis,:,:]+xtp[:,ti:(ti+1),]
        x3_mask = i2_formask[:,ti,:,0]!=0
        attn_output, attn4 = atten_layer(query=x3, value=x3, key=x3, return_attention_scores=True, attention_mask=x3_mask[:,tf.newaxis,tf.newaxis,:])
        if drop > 0:
            attn_output = drop1(attn_output)
        
        out1 = x3 + attn_output
        if is_batch == True:
            out1 = ln1(out1)
        
        ffn_output = pwffn_layer(out1)
        if drop > 0:
            ffn_output = drop2(ffn_output)
        
        out2 = out1 + ffn_output
        if is_batch == True:
            out2 = ln2(out2)
        
        x4 = out2        
        mask_t = tf.cast(x3_mask[:,:,tf.newaxis],tf.float32)
        enc_output2 = tf.math.multiply(x4,mask_t)
        x4 = tf.math.divide(K.sum(enc_output2, axis=1), K.sum(mask_t, axis=1)+tf.keras.backend.epsilon())   
        
        if ti==0:
            x = x4[:,tf.newaxis,:]
        else:
            x = tf.concat([x,x4[:,tf.newaxis,:]],axis=1)
    
    
    # 使用Transformer Encoder来处理嵌入后的图像和非图像数据
    # padding_mask_transformer = create_padding_mask(inputs=x, mask_value=0)  # 1451,1,1,94,64
    # padding_mask_transformer = create_padding_mask(inputs = inputs[0], mask_value=mask_value)
    # 在这里可以沿用前面的padding_mask，但是需要把0换成-9999
    # padding_mask_transformer = tf.where(tf.equal(padding_mask_cnn, 0), -9999, padding_mask_cnn)
    # encoder
    # 循环layern次，先将数据输入到多头注意力机制(MultiHeadAttention)中
    for i in range(layern):
        attn_output, attn4 = MultiHeadAttention(key_dim=units // n_head, num_heads=n_head)(query=x, value=x, key=x,
                                                                                           return_attention_scores=True,
                                                                                           attention_mask=padding_mask_cnn)
        # 然后通过残差连接和层归一化(LayerNormalization)进行处理
        if drop > 0:
            attn_output = Dropout(drop)(attn_output)
        
        out1 = x + attn_output        
        if is_batch == True:
            out1 = LayerNormalization(epsilon=1e-6)(out1)
        # 接着，通过前馈神经网络(point_wise_feed_forward_network)来对注意力输出进行处理。
        # ffn_output是通过将输入out1传递给point_wise_feed_forward_network函数，并将d_model设置为units，dff设置为units * 4来计算得出的。这意味着ffn_output是应用了一个点对点的前馈神经网络（Point-wise Feed Forward Network）到out1的结果。
        # ffn_output的形状将与输入out1的形状相同，为（batch_size，seq_len，d_model）。
        ffn_output = point_wise_feed_forward_network(units, units * 4)(out1)
        # 最后，再次进行残差连接和层归一化。
        if drop > 0:
            ffn_output = Dropout(drop)(ffn_output)
        out2 = out1 + ffn_output
        if is_batch == True:
            out2 = LayerNormalization(epsilon=1e-6)(out2)
        x = out2
    
    # 将Transformer Encoder的输出传递给一个全连接层(Dense)，并使用linear激活函数得到最终的输出。这个输出表示了模型对给定输入的预测。
    enc_output = x
    output = Dense(3, activation="linear")(enc_output) # [:,tf.newaxis,:]
    # for ii in range(n_out):
        # if ii==0:
            # output = Dense(1, activation="linear")(enc_output) # [:,tf.newaxis,:]
        # else:
            # output = tf.concat([output, Dense(1, activation="exponential")(enc_output)],axis=2) ## this does not work in for loop
    
    model = Model(inputs, output)
    return model


## *******************
## transformer_block
def transformer_block (queryx,x,padding_mask,units,n_head,reg=None,drop=0.1,is_batch=True,is_att_score=True):
    if is_att_score:
        attn_output, attn4 = MultiHeadAttention(key_dim=units//n_head, num_heads=n_head, kernel_regularizer=reg)(query=queryx, value=x, key=x,
            return_attention_scores=is_att_score, attention_mask=padding_mask)
    else:
        attn_output        = MultiHeadAttention(key_dim=units//n_head, num_heads=n_head, kernel_regularizer=reg)(query=queryx, value=x, key=x,
            return_attention_scores=is_att_score, attention_mask=padding_mask)
    
    if drop > 0:
        attn_output = Dropout(drop)(attn_output)
    
    out1 = queryx + attn_output
    if is_batch == True:
        out1 = LayerNormalization(epsilon=1e-6)(out1)
    
    ffn_output = point_wise_feed_forward_network(units, units * 4, reg=reg)(out1)
    if drop > 0:
        ffn_output = Dropout(drop)(ffn_output)
    
    out2 = out1 + ffn_output
    if is_batch == True:
        out2 = LayerNormalization(epsilon=1e-6)(out2)
    
    x = out2
    if is_att_score:
        return x, attn4
    else:
        return x


# layer_cnn=2; layern=3; units=32; n_times=input_images_train_norm3.shape[1]; 
# n_times_out=1; n_window=3; n_head=4; drop=0.001; n_out=3; 
# n_features=15; is_batch=True; mask_value=-9999.0; 
# is_zero_fill=True; doy_encoder=1; 
# is_rptv = True; is_sensor = False; is_rptv_sensor=False
# 
# inputs = (input_images_train_norm3 [:2,:,:,:,:], input_covariate_train_norm3[:2,:])
# inputs = (input_images_train_norm3 [:1,:,:,:,:], input_covariate_train_norm3[:1,:])
# inputs = input_images_train_norm3 [:2,:,:,:,:]
# a = model.predict(inputs)
# inputs[0] [0,:20,1,1,1]
# x  [0,:20,1,1,1]
# x  [0,:20,0,0,1]
# x  [0,:20,1]
# MODIS_BANDS_N = 9
# METER_BANDS_N = 6
def get_transformer_cnn3(layer_cnn=1, layern=2, units=64, n_times=80,
                         MODIS_BANDS_N=9, METER_BANDS_N=6, n_window=3, n_head=4, drop=0.1, n_out=3,
                         is_batch=True, mask_value=-9999.0, is_rptv=False, is_sensor=False):
    # 定义了一个Transformer-CNN模型，它是一个混合模型，同时接收图像数据和非图像数据作为输入，并将它们融合以生成最终的输出.这样设计的模型可以在处理复杂任务时融合多种数据源的信息，以提高模型的性能和泛化能力。
    # 定义了两个输入，一个是形状为(n_times, n_window, n_window, 7)的图像数据（inputs[0]）:
    # It appears that the model expects a 5-dimensional input where the first dimension represents the number of samples (None),
    # followed by three dimensions for the width, height, and channels (50, 50, 7),
    # and the last dimension represents another channel or feature (377).
    # 另一个是形状为(n_times, n_doy)的非图像数据（inputs[1]）。
    # inputs = [Input(shape=(n_times, n_window, n_window, 7)), Input(shape=(n_times, n_doy))]
    # inputs = [Input(shape=(n_times,n_window,n_window,n_bands)),Input(shape=(n_times,n_features))]
    inputs = Input(shape=(n_times, n_window, n_window, MODIS_BANDS_N + METER_BANDS_N))
    # x = inputs[0]
    # 创建用于训练数据 填充掩码的padding_mask和padding_mask3d 在这里之前应该确保x1完全为-9999时，x2或者其他的也应该是-9999
    # 这里还需要用x变量作mask 判断依据就是如果input_images_train_norm3第i条记录 第j天 第m个波段的值全为-9999，
    # 那么就将这条记录masked
    # padding_mask, padding_mask3d = create_padding_mask_any(inputs[1], mask_value)
    x_modis = inputs[:, :, :, :, :MODIS_BANDS_N]
    # 添加植被指数和光学指数之后；提取前 9 个波段和最后 2 个波段
    # x_modis_base = inputs[:, :, :, :,:MODIS_BANDS_N]  # 提取前 9 个波段
    # x_modis_last_two = inputs[:, :, :, :,-2:]  # 提取最后 2 个波段
    # # 拼接两个部分
    # x_modis = tf.concat((x_modis_base, x_modis_last_two), axis=4)  # 沿第 3 个维度拼接

    mask_modis, mask3d_modis = create_padding_mask_any(x_modis, mask_value)
    x_modis = x_modis * mask3d_modis[:, :, tf.newaxis, tf.newaxis, :]
    ## 处理非图像数据并进行嵌入(embedding)操作，将每个时间步的非图像数据转换为与图像数据具有相同维度的特征表示。
    ## *******************
    ## *******************
    # positional 位置编码的mask
    modis_d = layers.TimeDistributed(
        Conv2D(units, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal'))(x_modis)[:,
              :, 0, 0, :]
    x_doy = np.array(range(modis_d.shape[1]))[np.newaxis, :, np.newaxis]
    x_doy = (x_doy - x_doy.mean()) / x_doy.std()
    doy_d = Dense(units)(x_doy)
    # doy_d = Dense(units)(modis_d[:,:,:1])
    x_input_modis = modis_d + doy_d
    # embedding_sensor = Dense(units, activation='relu')
    # if is_sensor:
    #     x_input_modis = x_input_modis + embedding_sensor(xsensor)
    #
    #
    if is_rptv == True:
        x_meter = inputs[:, :, :, :, MODIS_BANDS_N:(METER_BANDS_N + MODIS_BANDS_N)]
        mask_meter, mask3d_meter = create_padding_mask_any(x_meter, mask_value)
        x_meter = x_meter[:, :, 1, 1, :] * mask3d_meter[:, :, :]
        meter_d = Dense(units)(x_meter)
        x_input_meter = meter_d + doy_d

        # print(xpp)
    # if doy_encoder == 1:
    # embedding_rptv = encoder_embed (units, encoder_layer=2)
    for i in range(layer_cnn):
        x_input_modis,attention_modis = transformer_block(x_input_modis, x_input_modis, mask_modis, units, n_head, drop=drop,
                                          is_batch=True, is_att_score=True);
        if is_rptv:
            x_input_meter,attention_RLM = transformer_block(x_input_meter, x_input_meter, mask_meter, units, n_head, drop=drop,
                                              is_batch=True, is_att_score=True);

    if is_rptv:
        x = x_input_modis + x_input_meter + doy_d
    else:
        x = x_input_modis

    # 使用Transformer Encoder来处理嵌入后的图像和非图像数据
    # padding_mask_transformer = create_padding_mask(inputs=x, mask_value=0)  # 1451,1,1,94,64
    # padding_mask_transformer = create_padding_mask(inputs = inputs[0], mask_value=mask_value)
    # 在这里可以沿用前面的padding_mask，但是需要把0换成-9999
    # encoder
    # 循环layern次，先将数据输入到多头注意力机制(MultiHeadAttention)中
    attention_scores = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)  # 初始化动态TensorArray

    for i in range(layern):
        x, attention_all = transformer_block(
            x, x, mask_modis, units, n_head, drop=drop, is_batch=True, is_att_score=True
        )
        attention_scores = attention_scores.write(i, attention_all)  # 写入当前层的注意力分数

    attention_scores = attention_scores.stack()  # 转换为形状为 (layern, ...) 的张量
    # method 1 simple linear
    # output = Dense(n_out, activation="linear")(enc_output)

    ## method 2
    enc_output = x
    for ii in range(n_out):
        if ii == 0:
            ouput0 = Dense(1, activation="linear")(enc_output)  # [:,tf.newaxis,:]
            output = ouput0
        else:
            ### ELU 没有神经元死亡的问题(ReLU Dying 问题是指当出现异常输入时，在反向传播中会产生大的梯度，这种大的梯度会导致神经元死亡和梯度消失)。
            output = tf.concat([output, Dense(1, activation="exponential")(enc_output)],
                               axis=2)  ## this does not work in for loop

    ## method 3 is not as good as I thought
    # 将Transformer Encoder的输出传递给一个全连接层(Dense)，并使用linear激活函数得到最终的输出。这个输出表示了模型对给定输入的预测。
    # enc_output1 = Dense(units, activation="relu")(x)
    # enc_output2 = Dense(units, activation="relu")(x)
    # enc_output3 = Dense(units, activation="relu")(x)
    # for ii in range(n_out):
    # if ii==0:
    # ouput1 = Dense(1, activation="linear"     )(enc_output1)
    # elif ii==1:
    # ouput2 = Dense(1, activation="exponential")(enc_output2)
    # else:
    # ouput3 = Dense(1, activation="exponential")(enc_output3)

    # output = tf.concat([ouput1, ouput2, ouput3],axis=2)

    model = Model(inputs, output)
    return model, attention_scores  # Return the model and attention scores
    # oupute = tf.keras.activations.exponential(ouput0)
    # output = tf.where(ouput0>0,oupute,oupute*(-1))
    # output = Dense(1, activation="linear")(enc_output) # [:,tf.newaxis,:]
    # output = ouput0*oupute


def get_transformer_cnn2(layer_cnn=1, layern=2, units=64, n_times=80,
                        MODIS_BANDS_N = 9,METER_BANDS_N = 6, n_window=3,n_head=4, drop=0.1,n_out=3,
                        is_batch=True, mask_value=-9999.0, is_rptv=False,is_sensor=False):
    # 定义了一个Transformer-CNN模型，它是一个混合模型，同时接收图像数据和非图像数据作为输入，并将它们融合以生成最终的输出.这样设计的模型可以在处理复杂任务时融合多种数据源的信息，以提高模型的性能和泛化能力。
    # 定义了两个输入，一个是形状为(n_times, n_window, n_window, 7)的图像数据（inputs[0]）:
    # It appears that the model expects a 5-dimensional input where the first dimension represents the number of samples (None),
    # followed by three dimensions for the width, height, and channels (50, 50, 7),
    # and the last dimension represents another channel or feature (377).
    # 另一个是形状为(n_times, n_doy)的非图像数据（inputs[1]）。
    # inputs = [Input(shape=(n_times, n_window, n_window, 7)), Input(shape=(n_times, n_doy))]
    # inputs = [Input(shape=(n_times,n_window,n_window,n_bands)),Input(shape=(n_times,n_features))]
    inputs = Input(shape=(n_times,n_window,n_window,MODIS_BANDS_N+METER_BANDS_N))
    # x = inputs[0]
    # 创建用于训练数据 填充掩码的padding_mask和padding_mask3d 在这里之前应该确保x1完全为-9999时，x2或者其他的也应该是-9999
    # 这里还需要用x变量作mask 判断依据就是如果input_images_train_norm3第i条记录 第j天 第m个波段的值全为-9999，
    # 那么就将这条记录masked
    # padding_mask, padding_mask3d = create_padding_mask_any(inputs[1], mask_value)
    x_modis = inputs[:,:,:,:,:MODIS_BANDS_N]
    # 添加植被指数和光学指数之后；提取前 9 个波段和最后 2 个波段
    # x_modis_base = inputs[:, :, :, :,:MODIS_BANDS_N]  # 提取前 9 个波段
    # x_modis_last_two = inputs[:, :, :, :,-2:]  # 提取最后 2 个波段
    # # 拼接两个部分
    # x_modis = tf.concat((x_modis_base, x_modis_last_two), axis=4)  # 沿第 3 个维度拼接

    mask_modis, mask3d_modis = create_padding_mask_any(x_modis, mask_value)
    x_modis = x_modis * mask3d_modis[:, :, tf.newaxis, tf.newaxis, :]
    ## 处理非图像数据并进行嵌入(embedding)操作，将每个时间步的非图像数据转换为与图像数据具有相同维度的特征表示。
    ## *******************
    ## *******************
    # positional 位置编码的mask
    modis_d = layers.TimeDistributed(Conv2D(units, kernel_size=(3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal'))(x_modis)[:,:,0,0,:]

    x_doy = np.array(range(modis_d.shape[1]))[np.newaxis,:,np.newaxis]
    x_doy = (x_doy-x_doy.mean()) /x_doy.std()
    doy_d = Dense(units)(x_doy)
    # doy_d = Dense(units)(modis_d[:,:,:1])
    x_input_modis = modis_d + doy_d
    # embedding_sensor = Dense(units, activation='relu')
    # if is_sensor:
    #     x_input_modis = x_input_modis + embedding_sensor(xsensor)
    #
    #
    if is_rptv==True:
        x_meter = inputs[:,:,:,:,MODIS_BANDS_N:(METER_BANDS_N+MODIS_BANDS_N)]
        mask_meter, mask3d_meter = create_padding_mask_any(x_meter, mask_value)
        x_meter = x_meter[:,:,1,1,:] * mask3d_meter[:, :, :]
        meter_d = Dense(units)(x_meter)
        x_input_meter = meter_d + doy_d    
    
    # print(xpp)
    # if doy_encoder == 1:
        # embedding_rptv = encoder_embed (units, encoder_layer=2)
    for i in range(layer_cnn): 
        x_input_modis = transformer_block (x_input_modis,x_input_modis,mask_modis,units,n_head,drop=drop,is_batch=True,is_att_score=False);
        if is_rptv:
            x_input_meter = transformer_block (x_input_meter,x_input_meter,mask_meter,units,n_head,drop=drop,is_batch=True,is_att_score=False);

    if is_rptv:
        x = x_input_modis+x_input_meter+ doy_d
    else:
        x = x_input_modis
    
    # 使用Transformer Encoder来处理嵌入后的图像和非图像数据
    # padding_mask_transformer = create_padding_mask(inputs=x, mask_value=0)  # 1451,1,1,94,64
    # padding_mask_transformer = create_padding_mask(inputs = inputs[0], mask_value=mask_value)
    # 在这里可以沿用前面的padding_mask，但是需要把0换成-9999
    # encoder
    # 循环layern次，先将数据输入到多头注意力机制(MultiHeadAttention)中
    for i in range(layern):
        x = transformer_block (x,x,mask_modis,units,n_head,drop=drop,is_batch=True,is_att_score=False);

    # method 1 simple linear 
    # output = Dense(n_out, activation="linear")(enc_output)
    
    ## method 2
    enc_output = x
    for ii in range(n_out):
        if ii==0:
            ouput0 = Dense(1, activation="linear")(enc_output) # [:,tf.newaxis,:]
            output = ouput0
        else:
            ### ELU 没有神经元死亡的问题(ReLU Dying 问题是指当出现异常输入时，在反向传播中会产生大的梯度，这种大的梯度会导致神经元死亡和梯度消失)。
            output = tf.concat([output, Dense(1, activation="exponential")(enc_output)],axis=2) ## this does not work in for loop

    ## method 3 is not as good as I thought 
    # 将Transformer Encoder的输出传递给一个全连接层(Dense)，并使用linear激活函数得到最终的输出。这个输出表示了模型对给定输入的预测。
    # enc_output1 = Dense(units, activation="relu")(x)
    # enc_output2 = Dense(units, activation="relu")(x)
    # enc_output3 = Dense(units, activation="relu")(x)
    # for ii in range(n_out):
        # if ii==0:
            # ouput1 = Dense(1, activation="linear"     )(enc_output1)
        # elif ii==1:
            # ouput2 = Dense(1, activation="exponential")(enc_output2)
        # else:
            # ouput3 = Dense(1, activation="exponential")(enc_output3)
    
    # output = tf.concat([ouput1, ouput2, ouput3],axis=2)
    
    model = Model(inputs, output)
    return model
            # oupute = tf.keras.activations.exponential(ouput0)
            # output = tf.where(ouput0>0,oupute,oupute*(-1))
            # output = Dense(1, activation="linear")(enc_output) # [:,tf.newaxis,:]
            # output = ouput0*oupute


# 函数 create_padding_mask 将输入张量 input 和 mask_value 作为参数。 它返回一个填充掩码张量。
# def create_padding_mask(inputs,mask_value=0):
#     # 通过检查输入张量的第一个通道 (inputs[:,:,0]) 中的每个元素是否不等于 mask_value 来创建填充掩码。
#     #  not_equal 函数将每个元素与 mask_value 进行比较，并返回与输入[:,:,0] 形状相同的布尔张量。
#     #  然后使用cast函数将布尔值转换为float32值，其中1表示该元素不等于mask_value，0表示该元素等于mask_value。
#     # seq = tf.cast(tf.math.not_equal(inputs[:,:,0], mask_value), tf.float32)
#     seq = tf.cast(tf.math.reduce_any(tf.math.not_equal(inputs, mask_value), axis=(2, 3, 4)), tf.float32)
#     new_seq = tf.where(tf.equal(seq, 0), tf.constant(-1e9, dtype=tf.float32), tf.cast(seq, tf.float32))
#     # 在这里应该注意将被填充的位置设置为-9999，还是和最开始CNN的mask一样，但是需要注意的是经过CNN的操作，x中被填充的地方变成了0
#     # 如果要进行mask 需要把输入变量x1 中为0的地方将padding_mask标记为-9999
#     # 生成的张量 seq 的形状为 (batch_size, seq_len) 并表示填充掩码。
#     # 最后，该函数返回填充掩码张量，并使用 [:, tf.newaxis, tf.newaxis, :] 添加两个附加维度。
#     # 生成的张量的形状为 (batch_size, 1, 1, seq_len)。 此格式通常用于在 3D 操作中应用填充掩码。
#     return new_seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# https://www.freecodecamp.org/news/the-ultimate-guide-to-recurrent-neural-networks-in-python/
## there must be at least one LSTM somewhere tested by HK on Nov 11, 2021
# 函数 get_regression_model 创建具有可定制层配置的回归模型。
def get_regression_model (n_features, layer_n=4, is_batch=True):
    # 它需要三个参数 n_features、layer_n 和 is_batch。 n_features指定输入特征的数量，layer_n指定模型中的层数，is_batch表示是否在每层之后使用批量归一化。
    # 函数内部有三个列表neuron_n_4layer、neuron_n_8layer和neuron_n_19layer，分别代表不同层配置的神经元数量。
    neuron_n_4layer  = [256, 512, 512]
    neuron_n_8layer  = [256, 256, 256, 512, 512,1024,1024]
    neuron_n_19layer = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512, 1024,1024]

    #  Neuron_n 变量默认初始化为neuron_n_4layer，但可以根据layer_n的值进行更新。
    ## parameters
    neuron_n = neuron_n_4layer
    if (layer_n<=4):
        neuron_n = neuron_n_4layer
    elif (layer_n<=8):
        neuron_n = neuron_n_8layer
    else:
        neuron_n = neuron_n_19layer
    # 该模型被初始化为顺序模型。 然后，循环执行layer_n-1次，其中每次迭代对应于向模型添加一个全连接层。
    ## build model
    model = Sequential()
    for i in range(layer_n-1):
        #  第一层的 kernel_initializer 设置为均值为 0 且标准差为 sqrt(2/neuron_n[i]) 的 RandomNormal 初始值设定项，
        #  后续层的 kernel_initializer 设置为具有相同参数的 TruncatedNormal 初始值设定项。 所有层的激活函数都设置为“relu”。
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=math.sqrt(2/neuron_n[i]) )
        if i==0:
            # initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=math.sqrt(2/neuron_n[i]) )
            model.add(Dense(neuron_n[i], activation='relu', kernel_initializer=initializer, input_shape=(n_features,)))
            # model.add(Dense(neuron_n[i], activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
        else:
            # initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=math.sqrt(2/neuron_n[i]) )
            model.add(Dense(neuron_n[i], activation='relu', kernel_initializer=initializer))
            # model.add(Dense(neuron_n[i], activation='relu', kernel_initializer='he_normal'))
        # 如果 is_batch 为 True，则在每个全连接层之后添加 BatchNormalization 层。
        if (is_batch):
            model.add(BatchNormalization())
    # 最后，添加一个具有 1 个单元的全连接层作为模型的输出层。
    model.add(Dense(1))
    return model


