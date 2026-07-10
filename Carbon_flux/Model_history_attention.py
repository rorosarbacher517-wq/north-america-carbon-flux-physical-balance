import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from keras import backend as K
from keras.callbacks import TensorBoard
# from tensorflow.keras import backend as K
# from tensorflow.keras.callbacks import TensorBoard
import logging


# logs_dir = "./fluxlogs" + 'logs'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=logs_dir, histogram_freq=1)

# 适应一个变量的准确度
class Maskedaccuracy(tf.keras.metrics.Metric):
    def __init__(self, mask_value=-9999, **kwargs):
        super(Maskedaccuracy, self).__init__(**kwargs)
        self.mask_value = mask_value
        self.total_mae = self.add_weight(name='total_mae', initializer='zeros')
        self.num_samples = self.add_weight(name='num_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        is_mask = tf.equal(y_true, self.mask_value)
        is_mask = tf.cast(is_mask, dtype=K.floatx())
        is_mask = 1 - is_mask
        y_true_masked = y_true * is_mask
        y_pred_masked = y_pred * is_mask
        masked_samples = tf.reduce_sum(is_mask, axis=list(range(1, K.ndim(y_true))))
        mae = K.abs(y_true_masked - y_pred_masked)
        masked_mae = mae * is_mask
        sum_masked_mae = tf.reduce_sum(masked_mae, axis=list(range(1, K.ndim(y_true_masked))))
        self.total_mae.assign_add(tf.reduce_sum(sum_masked_mae))
        self.num_samples.assign_add(tf.reduce_sum(masked_samples))

    def result(self):
        return self.total_mae / self.num_samples

    def reset_states(self):
        self.total_mae.assign(0.0)
        self.num_samples.assign(0.0)


# # 适应三个变量的
# class Maskedaccuracy(tf.keras.metrics.Metric):
#     def __init__(self, mask_value=-9999, **kwargs):
#         super(Maskedaccuracy, self).__init__(**kwargs)
#         self.mask_value = mask_value
#         self.total_accuracy = self.add_weight(name='total_accuracy', initializer='zeros')
#         self.num_samples = self.add_weight(name='num_samples', initializer='zeros')
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         is_mask = tf.equal(y_true, self.mask_value)
#         is_mask = tf.cast(is_mask, dtype=K.floatx())
#         is_mask = 1 - is_mask
#         y_true_masked = y_true * is_mask
#         y_pred_masked = y_pred * is_mask
#         masked_samples = tf.reduce_sum(is_mask, axis=list(range(1, K.ndim(y_true))))
#         accuracy_var = K.equal(tf.round(y_true_masked), tf.round(y_pred_masked))
#         accuracy_var = tf.cast(accuracy_var, dtype=K.floatx())
#         accuracy_var = accuracy_var * is_mask
#         sum_masked_accuracy = tf.reduce_sum(accuracy_var, axis=list(range(1, K.ndim(y_true_masked))))
#         self.total_accuracy.assign_add(tf.reduce_sum(sum_masked_accuracy))
#         self.num_samples.assign_add(tf.reduce_sum(masked_samples))
#
#     def result(self):
#         return self.total_accuracy / self.num_samples
#
#     def reset_states(self):
#         self.total_accuracy.assign(0.0)
#         self.num_samples.assign(0.0)

# def my_train_1schedule(model, input_images_train_norm3, train_datasety, epochs=10, start_rate=0.001, loss=np.nan, per_epoch=100,
# split_epoch=4, option=0, decay=1e-5,
# batch_size=32, validation_split=0.04, hold_epoch=0, reduce_epoch=False):


import train_test

## option is_layer_decay added on Jul 2023
## https://saturncloud.io/blog/how-to-set-layerwise-learning-rate-in-tensorflow/
# train_datasetx = input_images_train_norm3
# train_datasety = trainy
# # train_datasetx = [trainx1_transformer,trainx2_transformer]
# # train_datasety = trainy_transformer
# gaps_p = 0.1; gaps_n=1
# model= model; epochs=10;start_rate=0.001;loss=np.nan;per_epoch=100;split_epoch=4;option=0;decay=1e-5;
# batch_size=32;validation_split=0.04;hold_epoch=0;reduce_epoch=False; is_layer_decay=False; gaps_p=0.1
MODIS_BANDS_N = 9


def my_train_1schedule_time_drop(model, train_datasetx, train_datasety, epochs=10, start_rate=0.001, loss=np.nan,
                                 per_epoch=100, split_epoch=4, option=0, decay=1e-5,
                                 batch_size=32, validation_split=0.04, hold_epoch=0, reduce_epoch=False, gaps_p=0.1,
                                 gaps_n=0):
    momentum = 0.9  # good for RMSprop not good for RMSprop but OK for Adam
    print_str = "\nmomentum=" + str(momentum) + "\tlearning rate=" + str(start_rate) + "  decay=" + str(decay)
    print(print_str);
    logging.info(print_str)
    ##*****************************************************************************************************
    ## random mask with tensor
    # x1 = trainx_transformer[90,:,:]; y1 = trainy_transformer[90,:,:];
    # x1 = trainx_transformer[0,:,:]; y1 = trainy_transformer[0];
    # ONES  = tf.ones ([train_datasetx.shape[1],train_datasetx.shape[2]])
    # ZEROS = tf.zeros([train_datasetx.shape[1],train_datasetx.shape[2]])
    BNAD_N = train_datasetx.shape[-1]
    train_datasety = train_datasety.astype(np.float32)
    # x1 = X_train_sub[0,:,:,:,:]; y1 = y_train_sub[0,:]
    ONES = tf.ones([train_datasetx.shape[1], 1, 1, BNAD_N - MODIS_BANDS_N])
    ZEROS = tf.zeros([train_datasetx.shape[1], 1, 1, BNAD_N - MODIS_BANDS_N])
    gaps_n = tf.cast(gaps_n, tf.int32)

    def random_mask(x1, y1):
        mask_value = -9999.0
        # x_mask = x1[:,0]!=mask_value
        x_mask = tf.math.reduce_any(tf.math.not_equal(x1[:, :, :, 0], mask_value), axis=(1, 2))
        input_length = tf.math.reduce_sum(
            tf.cast(x_mask, tf.int32))  # !!!!!!!!!!!!! # a bug fixed on Mar 4, 2013 !!!!!!!!!!!!!!!!!!!!!!
        delete_n_max = tf.cast((tf.cast(input_length, tf.float32) * gaps_p + 0.5), tf.int32)
        delete_n_max = tf.math.minimum(input_length - 1, delete_n_max)  # set max as input_length-1
        # if gaps_n >0
        delete_n = tf.cond(gaps_n > 0, lambda: gaps_n, lambda: delete_n_max)

        # delete_n = tf.math.maximum(1,delete_n)
        # delete_n = tf.random.uniform(shape=[], minval=0, maxval=delete_n_max, dtype=tf.int32)
        # delete_n = tf.cond(delete_n_max>0,lambda: tf.random.uniform(shape=[], minval=0, maxval=delete_n_max, dtype=tf.int32),lambda: tf.constant(0) )

        # if (delete_n>0).numpy():
        def get_x3(delete_n):
            # if bool(delete_n>0):
            delete_n = tf.math.maximum(1, delete_n)
            num_indices_to_keep = input_length - delete_n
            # create uniform distribution over the sequence
            uniform_distribution = tf.random.uniform(shape=[input_length], minval=0, maxval=None, dtype=tf.float32,
                                                     seed=None, name=None)
            # grab the indices of the greatest num_words_to_drop values from the distibution
            _, indices_to_keep = tf.nn.top_k(uniform_distribution, delete_n)
            sorted_indices_to_keep = tf.sort(indices_to_keep)
            # gather indices from the input array using the filtered actual array
            deleted_y = tf.gather(tf.where(x_mask)[:, 0], sorted_indices_to_keep)
            deleted_y
            ## **********************************
            ## convert random selected to 2d
            ## https://www.tensorflow.org/guide/tensor_slicing#insert_data_into_tensors
            updates = tf.ones([delete_n])
            t9 = tf.SparseTensor(indices=tf.reshape(deleted_y, [delete_n, 1]), values=updates,
                                 dense_shape=[x1.shape[0]])

            t10 = tf.sparse.to_dense(t9)
            t10_reverse = tf.cast(tf.math.not_equal(t10, 1), tf.float32)  # TensorShape([80])
            y_multi = t10[:, tf.newaxis, tf.newaxis, tf.newaxis]
            x_multi = t10_reverse[:, tf.newaxis, tf.newaxis, tf.newaxis]
            x_add = tf.math.multiply(y_multi, mask_value)
            # *******
            # fixed on Mar 16 2023 to only allow band ref filled + ## important to only allow reflectance filled but not doy, x and y and sensor
            # x_multi = tf.repeat(x_multi, repeats=MODIS_BANDS_N, axis=-1);
            aa = tf.repeat(x_multi, repeats=MODIS_BANDS_N, axis=-1);
            x_multi = tf.concat([aa, ONES], axis=-1)
            aa = tf.repeat(x_add, repeats=MODIS_BANDS_N, axis=-1);
            x_add = tf.concat([aa, ZEROS], axis=-1)
            ## **********************************
            x3 = tf.math.multiply(x1, x_multi) + x_add
            return x3

            # x2 = get_x3(delete_n)

        x3 = tf.cond(delete_n > 0, lambda: get_x3(delete_n), lambda: x1)
        y3 = y1
        # y3 = tf.math.multiply(y1,y_multi)+y_add
        return x3, y3

    if validation_split == 0:
        X_train_sub, y_train_sub = train_datasetx, train_datasety
    else:
        X_train_sub, y_train_sub, X_validation, y_validation, training_index, validation_index = \
            train_test.random_split_train_validation(X_train=train_datasetx, y_train=train_datasety,
                                                     pecentage=validation_split)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    with tf.device("/cpu:0"):  ## this is added on Aug 21, 2021 to handle tensorflow 2.6 by forcing the prefetch on CPU
        if validation_split != 0:
            val_batch = min(y_validation.shape[0], batch_size)  # this bug is fixed on Mar 5, 2023

            print("validation batzh size " + str(val_batch))
            validation_ds = tf.data.Dataset.from_tensor_slices((X_validation, y_validation)).batch(val_batch)
            # validation_ds = tf.data.Dataset.from_tensor_slices( ({"input_1": X_validation[0], "input_2": X_validation[1]}, y_validation)).batch(val_batch)
            validation_ds = validation_ds.with_options(options)

        train_n = X_train_sub.shape[0]
        train_ds = tf.data.Dataset.from_tensor_slices((X_train_sub, y_train_sub)).shuffle(train_n + 1).map(
            lambda x, y: random_mask(x, y)).batch(batch_size)
        # train_ds = tf.data.Dataset.from_tensor_slices( ({"input_1": X_train_sub[0], "input_2": X_train_sub[1]}, y_train_sub)).shuffle(train_n+1).map(lambda x, y: random_mask(x,y)).batch(batch_size)
        # train_ds = tf.data.Dataset.from_tensor_slices( (X_train_sub, y_train_sub)).shuffle(train_n+1).batch(batch_size)
        train_ds = train_ds.with_options(options)

    reduced_e = 10
    reduced_e = 5
    # reduced_e = 15
    if reduce_epoch:
        final_epochs = (epochs - reduced_e) if epochs > reduced_e else epochs
    else:
        final_epochs = epochs

    # with strategy.scope():
    total_steps = per_epoch * epochs
    warmup_steps = per_epoch * split_epoch
    hold_steps = per_epoch * hold_epoch

    schedule_one = WarmUpCosineDecay(start_lr=0.0, target_lr=start_rate, warmup_steps=warmup_steps,
                                     total_steps=total_steps, hold=hold_steps)

    if option == -1:
        print('tfa.optimizers.SGD ');
        optimizer = tf.keras.optimizers.SGD(learning_rate=schedule_one, momentum=momentum);
    elif option == 0:
        print('tfa.optimizers.RMSprop ');
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=schedule_one, rho=momentum);
    elif option == 2:
        print('tfa.optimizers.Adam ');
        optimizer = tf.keras.optimizers.Adam(learning_rate=schedule_one, beta_1=momentum, beta_2=0.999, epsilon=1e-07)
    else:
        # print ('tfa.optimizers.AdamW ');  optimizer=tf.keras.optimizers.AdamW  (learning_rate=schedule_one, weight_decay=decay,beta_1=momentum)
        print("decay = {:5.4f}".format(decay))
        print('tfa.optimizers.AdamW ');
        optimizer = tfa.optimizers.AdamW(weight_decay=decay, learning_rate=schedule_one, beta_1=momentum)

    # model.compile(optimizer=optimizer, loss=loss, metrics=[Maskedaccuracy()])
    model.compile(
        optimizer=optimizer,
        loss={'predictions': loss, 'attentions': None},  # 不对 attentions 求 loss
        loss_weights={'predictions': 1.0, 'attentions': 0.0},
        metrics={'predictions': [Maskedaccuracy()]}
    )
    import socket
    if validation_split != 0:
        model_history = model.fit(train_ds, validation_data=validation_ds, epochs=final_epochs, verbose=2,
                                  callbacks=[LRTensorBoard(log_dir="./tmp/tb_log/" + socket.gethostname())])
    else:
        model_history = model.fit(train_ds, epochs=final_epochs, verbose=2,
                                  callbacks=[LRTensorBoard(log_dir="./tmp/tb_log/" + socket.gethostname())])

    np.set_printoptions(suppress=True)  # turn off scientific notation
    np.set_printoptions(precision=3)
    # for lri in
    print(np.array(model_history.history['lr']) * 100)
    print(model_history.history['lr'])
    return model_history


## https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
def lr_warmup_cosine_decay(global_step, warmup_steps, hold=0, total_steps=0, target_lr=1e-3, start_lr=0.0):
    # Cosine decay
    # There is no tf.pi so we wrap np.pi as a TF constant
    global_step_tf = tf.cast(global_step, tf.float32)
    learning_rate = 0.5 * target_lr * (1 + tf.cos(
        tf.constant(np.pi) * (global_step_tf - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = tf.cast(target_lr * (global_step / warmup_steps), tf.float32)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold, learning_rate, target_lr)

    learning_rate = tf.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return tf.cast(learning_rate, tf.float32)


## https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    # 首先使用余弦函数（tf.cos()）实现余弦衰减，然后使用tf.cast()函数将学习率转换为tf.float32类型，
    # 最后，使用tf.where()函数根据global_step的值来选择是否继续使用热启动学习率或余弦衰减学习率
    def __init__(self, start_lr, target_lr, warmup_steps, total_steps, hold):
        super().__init__()
        self.start_lr = start_lr  # 初始学习率
        self.target_lr = target_lr  # 目标学习率
        self.warmup_steps = warmup_steps  # 学习率热启动的步数，即在这些步数内学习率会逐渐增加
        self.total_steps = total_steps  # 总步数，代表训练的总迭代数
        self.hold = hold  # 学习率在达到目标学习率后保持恒定的步数

    def __call__(self, step):
        # 根据输入的step值，通过调用lr_warmup_cosine_decay函数计算学习率lr。这个函数实现了余弦衰减和学习率热启动的逻辑。
        lr = lr_warmup_cosine_decay(global_step=step, total_steps=self.total_steps, warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr, hold=self.hold)
        # # 最后，通过使用tf.where函数来判断是否超过了总步数，超过则返回学习率为0.0，否则返回计算得到的学习率
        return tf.where(step > self.total_steps, 0.0, lr, name="learning_rate")


# 定义了一个名为LRTensorBoard的自定义TensorBoard回调类，它继承自TensorBoard类。
# # 在每个训练周期结束时记录学习率的值到TensorBoard日志文件中，以便可视化学习率的变化。
class LRTensorBoard(TensorBoard):
    # 在构造函数__init__中，它接收一个log_dir参数和其他可选参数 **kwargs，
    # 并通过调用父类的构造函数super().__init__(log_dir=log_dir, **kwargs)来初始化TensorBoard回调。
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)

    # # 在on_epoch_end方法中，回调函数会在每个训练周期结束时调用。
    def on_epoch_end(self, epoch, logs=None):
        # 在这个方法中，它首先初始化logs为一个空字典（如果logs参数没有提供的话）
        logs = logs or {}
        # 获取当前的优化器optimizer和学习率current_lr
        optimizer = self.model.optimizer
        # 检查学习率是否是tf.keras.optimizers.schedules.LearningRateSchedule类型的实例
        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            # 如果是，则调用lr(optimizer.iterations)来计算当前学习率
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            # 否则，直接获取optimizer.lr的值作为当前学习率。
            current_lr = optimizer.lr
        # 使用K.eval()函数将当前学习率的值转换为TensorFlow张量的计算结果，并将其添加到logs字典中，键名为'lr'
        logs.update({'lr': K.eval(current_lr)})
        # 通过调用父类的super().on_epoch_end(epoch, logs)来执行TensorBoard回调的on_epoch_end方法，将日志写入TensorBoard日志文件中。
        # 将日志写入log日志文件中
        logging.info(logs)
        super().on_epoch_end(epoch, logs)


def my_train_1schedule(model, input_images_train_norm3, train_datasety, epochs=10, start_rate=0.001, loss=np.nan,
                       per_epoch=100,
                       split_epoch=4, option=0, decay=1e-5,
                       batch_size=32, validation_split=0.04, hold_epoch=0, reduce_epoch=False):
    momentum = 0.8  # 动量
    # 创建了一个字符串print_str，用于打印和记录有关动量、学习率和衰减的信息
    print_str = "momentum=" + str(momentum) + "\tlearning rate=" + str(start_rate) + "  decay=" + str(decay)
    # 调用print函数和logging.info函数将print_str打印和记录到日志中。
    print(print_str);
    logging.info(print_str)
    ##*****************************************************************************************************
    ## warm up
    # warm_up = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=start_rate/split_epoch/per_epoch,decay_steps=split_epoch*per_epoch,end_learning_rate=start_rate,name='Decay_linear')
    total_steps = per_epoch * epochs  # 表示总步数，即每个周期内的总迭代次数乘以总周期数
    warmup_steps = per_epoch * split_epoch  # 表示热启动的步数,在这些步数内学习率会逐渐增加
    hold_steps = per_epoch * hold_epoch  # hold_steps表示学习率保持恒定的步数
    # 学习率衰减策略对象schedule_one 控制训练过程中学习率的变化。
    schedule_one = WarmUpCosineDecay(start_lr=0.0, target_lr=start_rate, warmup_steps=warmup_steps,
                                     total_steps=total_steps, hold=hold_steps)
    if option == 0:
        print('tfa.optimizers.RMSprop');
        # 实例化一个tf.keras.optimizers.RMSprop对象，将衰减率decay、学习率schedule_one和动量rho设置为给定的数值。
        optimizer = tf.keras.optimizers.RMSprop(decay=decay, learning_rate=schedule_one, rho=momentum);
    elif option == 1:
        # 实例化一个tf.keras.optimizers.Adam对象，将学习率schedule_one、一阶矩估计的动量beta_1、二阶矩估计的动量beta_2和epsilon设置为给定的数值。
        print('tfa.optimizers.Adam ');
        optimizer = tf.keras.optimizers.Adam(learning_rate=schedule_one, beta_1=momentum, beta_2=0.999, epsilon=1e-07)
    else:
        # 实例化一个tfa.optimizers.AdamW对象，将权重衰减因子weight_decay、学习率schedule_one和动量beta_1设置为给定的数值。
        print('tfa.optimizers.AdamW ');
        optimizer = tfa.optimizers.AdamW(weight_decay=decay, learning_rate=schedule_one, beta_1=momentum)

    # 设置了模型的编译参数，包括优化器（optimizer）、损失函数（loss）和评估指标（metrics）。使用model.compile函数将这些参数应用于模型。
    model.compile(optimizer=optimizer, loss=loss, metrics=[Maskedaccuracy()])
    # model_history = model.fit(train_dataset, validation_data=validation_dataset, epochs=split_epoch, verbose=2)
    if reduce_epoch:
        # 如果reduce_epoch为True，则将最终的训练周期数减去10，但不会小于10。
        final_epochs = (epochs - 10) if epochs > 10 else epochs
    else:
        # 否则，最终的训练周期数等于初始设定的周期数。
        final_epochs = epochs
    # 然后，根据是否设置了验证集的比例，使用适当的参数调用model.fit方法来训练模型。同时，使用LRTensorBoard回调函数来记录训练过程中的学习率信息。
    # 将输入数据转为tensor
    # 将numpy数组转换为tf.Tensor并创建tf.data.Dataset对象
    input_images_train_norm3 = tf.convert_to_tensor(input_images_train_norm3, dtype=tf.float32)
    # traintime = tf.convert_to_tensor(traintime, dtype=tf.float32)
    # train_datasetx = [input_images_train_norm3, traintime]
    train_datasetx = input_images_train_norm3
    train_datasety = tf.convert_to_tensor(train_datasety, dtype=tf.float32)
    if validation_split > 0:
        # 训练集数据(train_datasetx和train_datasety)，验证集，最终的训练周期数(final_epochs)、输出详细信息的级别(verbose)、批量大小(batch_size)以及一个回调函数LRTensorBoard来记录训练过程中的学习率和其他信息。
        model_history = model.fit(x=train_datasetx, y=train_datasety, validation_split=validation_split,
                                  epochs=final_epochs,
                                  verbose=2, batch_size=batch_size, callbacks=[LRTensorBoard(log_dir="./tmp/tb_log")])
        # model_history = model.fit(x=train_datasetx, y=train_datasety, validation_split=0.1, epochs=final_epochs,
        #                           verbose=2, batch_size=batch_size, callbacks=[tensorboard_callback])
    else:
        model_history = model.fit(x=train_datasetx, y=train_datasety, epochs=final_epochs, verbose=2,
                                  batch_size=batch_size, callbacks=[LRTensorBoard(log_dir="./tmp/tb_log")])
        # model_history = model.fit(x=train_datasetx, y=train_datasety, epochs=final_epochs, verbose=2,
        #                           batch_size=batch_size, callbacks=[tensorboard_callback])
    # print(model_history )
    return model_history

