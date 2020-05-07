import os
import random
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.python.framework import graph_util
from tensorflow.python.training.moving_averages import assign_moving_average
import data_loader


class ModelDetect:
    def __init__(self,
                 model_detect_dir,
                 model_detect_pb_file,
                 LEARNING_RATE_BASE,
                 TRAINING_STEPS,
                 VALID_FREQ,
                 LOSS_FREQ,
                 KEEP_NEAR,
                 KEEP_FREQ,
                 anchor_heights,
                 MOMENTUM,
                 dir_results_valid,
                 threshold,
                 model_detect_name,
                 rnn_size,
                 fc_size,
                 keep_prob):
        self.model_detect_dir = model_detect_dir
        self.model_detect_pb_file = model_detect_pb_file
        self.pb_file = os.path.join(model_detect_dir, model_detect_pb_file)
        self.sess_config = tf.ConfigProto()
        self.is_train = False
        self.graph = None
        self.sess = None
        self.learning_rate_base = LEARNING_RATE_BASE
        self.train_steps = TRAINING_STEPS
        self.valid_freq = VALID_FREQ
        self.loss_freq = LOSS_FREQ
        self.keep_near = KEEP_NEAR
        self.keep_freq = KEEP_FREQ
        self.anchor_heights = anchor_heights
        self.MOMENTUM = MOMENTUM
        self.dir_results_valid = dir_results_valid
        self.threshold = threshold
        self.model_detect_name = model_detect_name
        self.rnn_size = rnn_size
        self.fc_size = fc_size
        self.keep_prob = keep_prob

    def prepare_for_prediction(self, pb_file_path=None):
        """
        加载计算图
        :param pb_file_path: pb文件
        :return:
        """
        if pb_file_path == None:
            pb_file_path = self.pb_file

        if not os.path.exists(pb_file_path):
            print('ERROR: %s NOT exists, when load_pb_for_predict()' % pb_file_path)
            return -1

        self.graph = tf.Graph()

        # 从pb文件导入计算图
        with self.graph.as_default():
            with open(pb_file_path, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")

            self.x = self.graph.get_tensor_by_name('x-input:0')
            self.w = self.graph.get_tensor_by_name('w-input:0')

            self.rnn_cls = self.graph.get_tensor_by_name('rnn_cls:0')
            self.rnn_ver = self.graph.get_tensor_by_name('rnn_ver:0')
            self.rnn_hor = self.graph.get_tensor_by_name('rnn_hor:0')

        print('graph loaded for prediction')
        self.sess = tf.Session(graph=self.graph, config=self.sess_config)

    def predict(self, img_file, out_dir=None):
        """
        :param img_file: 图像路径. [str]
        :param out_dir: 输出保存路径. [str]
        :return:
        """
        # 加载图像
        img = Image.open(img_file)

        # 图片预处理
        # img_data = data_loader.mean_gray(img_data)
        # img_data = data_loader.two_value_binary(img_data)
        # img_data = data_loader.convert2rgb(img_data)

        # 对图像进行放缩
        img_size = img.size  # (width, height)
        im_size_min = np.min(img_size[0:2])
        im_size_max = np.max(img_size[0:2])
        im_scale = float(600) / float(im_size_min)
        if np.round(im_scale * im_size_max) > 800:
            im_scale = float(800) / float(im_size_max)
        width = int(img_size[0] * im_scale)
        height = int(img_size[1] * im_scale)
        img = img.resize((width, height), Image.ANTIALIAS)
        # re_im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

        # 对图像进行标准化
        img_data = np.array(img, dtype=np.float32) / 255
        try:
            img_data = [img_data[:, :, 0:3]]  # rgba
        except:
            img_data = [img_data[:, :, 0:2]]  # rgb
        w_arr = np.array([width], dtype=np.int32)

        # 开始预测
        with self.graph.as_default():
            feed_dict = {self.x: img_data, self.w: w_arr}
            r_cls, r_ver, r_hor = self.sess.run([self.rnn_cls, self.rnn_ver, self.rnn_hor], feed_dict)
            text_bbox, conf_bbox = data_loader.trans_results(r_cls, r_ver, r_hor, \
                                                             self.anchor_heights, self.threshold)
            # refinement
            conn_bbox = data_loader.do_nms_and_connection(text_bbox, conf_bbox)

            if out_dir == None:
                return conn_bbox, text_bbox, conf_bbox

            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            # 绘制anchor文本线
            filename = os.path.basename(img_file)
            basename, _ = os.path.splitext(filename)
            file_target = os.path.join(out_dir, 'predicted_' + basename + '.png')
            img_target = Image.fromarray(np.uint8(img_data[0] * 255))  # .convert('RGB')
            img_target.save(file_target)
            data_loader.draw_text_boxes(file_target, text_bbox)

            # 绘制多个anchor连接后的文本线
            file_target = os.path.join(out_dir, 'connected_' + basename + '.png')
            img_target = Image.fromarray(np.uint8(img_data[0] * 255))  # .convert('RGB')
            img_target.save(file_target)
            data_loader.draw_text_boxes(file_target, conn_bbox)

            return conn_bbox, text_bbox, conf_bbox

    def create_graph_all(self, training):
        """
        创建计算图
        :param training: 参数是否可训练. [boolean]
        :return:
        """
        self.is_train = training
        self.graph = tf.Graph()

        with self.graph.as_default():
            # 初始化变量
            self.x = tf.placeholder(tf.float32, (1, None, None, 3), name='x-input')
            self.w = tf.placeholder(tf.int32, (1,), name='w-input')  # width
            self.t_cls = tf.placeholder(tf.float32, (None, None, None), name='c-input')
            self.t_ver = tf.placeholder(tf.float32, (None, None, None), name='v-input')
            self.t_hor = tf.placeholder(tf.float32, (None, None, None), name='h-input')

            # 卷积层，结合resnet结构
            self.conv_feat, self.seq_len = self.conv_feat_layers(self.x, self.w, self.is_train)

            # BI_LSTM + 全连接层
            self.rnn_cls, self.rnn_ver, self.rnn_hor = self.rnn_detect_layers(self.conv_feat,
                                                                              self.seq_len,
                                                                              len(self.anchor_heights))

            # 模型的损失函数
            self.loss = self.detect_loss(self.rnn_cls,
                                         self.rnn_ver,
                                         self.rnn_hor,
                                         self.t_cls,
                                         self.t_ver,
                                         self.t_hor)

            # 设置优化函数
            self.global_step = tf.train.get_or_create_global_step()
            self.learning_rate = tf.get_variable("learning_rate", shape=[], dtype=tf.float32, trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.MOMENTUM)
            grads_applying = optimizer.compute_gradients(self.loss)
            self.train_op = optimizer.apply_gradients(grads_applying, global_step=self.global_step)

            if self.is_train:
                print('graph defined for training')
            else:
                print('graph defined for validation')

    def train_and_valid(self, data_train, data_valid):
        """
        训练模型
        :param data_train: 训练集图像路径列表. [list]
        :param data_valid: 测试集图像路径列表. [list]
        :return:
        """
        # 创建模型存储路径
        if not os.path.exists(self.model_detect_dir):
            os.mkdir(self.model_detect_dir)

        # 构建计算图
        self.create_graph_all(training=True)

        # 加载和训练模型
        with self.graph.as_default():
            saver = tf.train.Saver()
            with tf.Session(config=self.sess_config) as sess:
                # 初始化变量
                tf.global_variables_initializer().run()
                sess.run(tf.assign(self.learning_rate, tf.constant(self.learning_rate_base, dtype=tf.float32)))

                # 加载模型
                ckpt = tf.train.get_checkpoint_state(self.model_detect_dir)

                # 加载模型
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                # 开始训练
                print('begin to train ...')
                start_time = time.time()
                begin_time = start_time
                step = sess.run(self.global_step)
                train_step_half = int(self.train_steps * 0.5)
                train_step_quar = int(self.train_steps * 0.75)

                while step < self.train_steps:
                    # 训练次数达到总的迭代次数的一半时，将学习率设置为原来的0.1，
                    # 当训练次数达到总的迭代次数的3/4时，将学习率设置为原来的0.01
                    if step == train_step_half:
                        sess.run(
                            tf.assign(self.learning_rate, tf.constant(self.learning_rate_base / 10, dtype=tf.float32)))
                    if step == train_step_quar:
                        sess.run(
                            tf.assign(self.learning_rate, tf.constant(self.learning_rate_base / 100, dtype=tf.float32)))

                    # 保存和验证模型
                    if (step + 1) % self.valid_freq == 0:
                        # 保存模型
                        print('save model to ckpt ...')
                        saver.save(sess, os.path.join(self.model_detect_dir, self.model_detect_name),
                                   global_step=step)

                        # 验证模型
                        print('validating ...')
                        model_v = ModelDetect(self.model_detect_dir,
                                              self.model_detect_pb_file,
                                              self.learning_rate_base,
                                              self.train_steps,
                                              self.valid_freq,
                                              self.loss_freq,
                                              self.keep_near,
                                              self.keep_freq,
                                              self.anchor_heights,
                                              self.MOMENTUM,
                                              self.dir_results_valid,
                                              self.threshold,
                                              self.model_detect_name,
                                              self.rnn_size,
                                              self.fc_size,
                                              1.0)
                        model_v.validate(data_valid, step)

                    # 从训练集中随机抽选一张照片
                    img_file = random.choice(data_train)
                    if not os.path.exists(img_file):
                        print('image_file: %s NOT exist' % img_file)
                        continue

                    # 获取该图像的文本线文档路径
                    txt_file = data_loader.get_target_txt_file(img_file)
                    if not os.path.exists(txt_file):
                        print('label_file: %s NOT exist' % txt_file)
                        continue

                    # 加载图像，并获取对应的真实标签
                    img_data, feat_size, target_cls, target_ver, target_hor = \
                        data_loader.get_image_and_targets(img_file, txt_file, self.anchor_heights)

                    # 开始训练
                    img_size = img_data[0].shape  # height, width, channel
                    w_arr = np.array([img_size[1]], dtype=np.int32)

                    feed_dict = {self.x: img_data,
                                 self.w: w_arr,
                                 self.t_cls: target_cls,
                                 self.t_ver: target_ver,
                                 self.t_hor: target_hor}

                    _, loss_value, step, lr = sess.run([self.train_op, self.loss, self.global_step, self.learning_rate],
                                                       feed_dict)

                    if step % self.loss_freq == 0:
                        curr_time = time.time()
                        print('step: %d, loss: %g, lr: %g, sect_time: %.1f, total_time: %.1f, %s' %
                              (step, loss_value, lr,
                               curr_time - begin_time,
                               curr_time - start_time,
                               os.path.basename(img_file)))
                        begin_time = curr_time

    def validate(self, data_valid, step):
        """
        模型验证函数
        :param data_valid: 验证集图像路径列表. [list]
        :param step: 当前迭代的次数. [int]
        :return:
        """
        # 判断验证集路径是否存在
        if not os.path.exists(self.dir_results_valid):
            os.mkdir(self.dir_results_valid)

        # 初始化计算图
        self.create_graph_all(training=False)

        with self.graph.as_default():
            saver = tf.train.Saver()
            with tf.Session(config=self.sess_config) as sess:
                # 初始化全局变量
                tf.global_variables_initializer().run()

                # 加载模型
                ckpt = tf.train.get_checkpoint_state(self.model_detect_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                # 将变量转化为常数，并保存到pb文件
                constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                           output_node_names=['rnn_cls', 'rnn_ver',
                                                                                              'rnn_hor'])
                with tf.gfile.FastGFile(self.pb_file, mode='wb') as f:
                    f.write(constant_graph.SerializeToString())

                # 开始预测
                NumImages = len(data_valid)
                curr = 0
                for img_file in data_valid:
                    print(img_file)
                    # 获取当前图像的文本线txt文档的存储路径
                    txt_file = data_loader.get_target_txt_file(img_file)

                    # 获取当前图像的像素矩阵、feature map维度以及三个分支的标签
                    img_data, feat_size, target_cls, target_ver, target_hor = \
                        data_loader.get_image_and_targets(img_file, txt_file, self.anchor_heights)

                    # 当前图像的尺寸
                    img_size = img_data[0].shape  # height, width, channel
                    w_arr = np.array([img_size[1]], dtype=np.int32)

                    feed_dict = {self.x: img_data,
                                 self.w: w_arr,
                                 self.t_cls: target_cls,
                                 self.t_ver: target_ver,
                                 self.t_hor: target_hor}

                    # 获取预测到的标签和损失值
                    r_cls, r_ver, r_hor, loss_value = sess.run([self.rnn_cls, self.rnn_ver, self.rnn_hor, self.loss],
                                                               feed_dict)

                    curr += 1
                    print('curr: %d / %d, loss: %f' % (curr, NumImages, loss_value))

                    # 将相对坐标转化为原始图像的绝对坐标，获取预测到的文本线坐标和分数
                    text_bbox, conf_bbox = data_loader.trans_results(r_cls,
                                                                     r_ver,
                                                                     r_hor,
                                                                     self.anchor_heights,
                                                                     self.threshold)

                    # 在图像上绘制文本线，并保存
                    filename = os.path.basename(img_file)
                    file_target = os.path.join(self.dir_results_valid, str(step) + '_predicted_' + filename)
                    img_target = Image.fromarray(np.uint8(img_data[0] * 255))  # .convert('RGB')
                    img_target.save(file_target)
                    data_loader.draw_text_boxes(file_target, text_bbox)

                    # 移除之前验证的文件
                    id_remove = step - self.valid_freq * self.keep_near
                    if id_remove % self.keep_freq:
                        file_temp = os.path.join(self.dir_results_valid, str(id_remove) + '_predicted_' + filename)
                        if os.path.exists(file_temp): os.remove(file_temp)

                print('validation finished')

    def norm_layer(self, x, train, eps=1e-05, decay=0.9, affine=True, name=None):
        """
        批标准化
        :param x:输入. [tensor]
        :param train: 是否可训练. [boolean]
        :param eps:
        :param decay:
        :param affine:
        :param name:
        :return:
        """
        with tf.variable_scope(name, default_name='batch_norm'):
            params_shape = [x.shape[-1]]
            batch_dims = list(range(0, len(x.shape) - 1))
            moving_mean = tf.get_variable('mean', params_shape,
                                          initializer=tf.zeros_initializer(),
                                          trainable=False)
            moving_variance = tf.get_variable('variance', params_shape,
                                              initializer=tf.ones_initializer(),
                                              trainable=False)

            def mean_var_with_update():
                # 计算均值和方差
                batch_mean, batch_variance = tf.nn.moments(x, batch_dims, name='moments')
                # 更新moving_mean和moving_variance
                with tf.control_dependencies([assign_moving_average(moving_mean, batch_mean, decay),
                                              assign_moving_average(moving_variance, batch_variance, decay)]):
                    return tf.identity(batch_mean), tf.identity(batch_variance)

            if train:
                mean, variance = mean_var_with_update()
            else:
                mean, variance = moving_mean, moving_variance

            if affine:
                beta = tf.get_variable('beta', params_shape,
                                       initializer=tf.zeros_initializer(),
                                       trainable=True)
                gamma = tf.get_variable('gamma', params_shape,
                                        initializer=tf.ones_initializer(),
                                        trainable=True)
                x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
            else:
                x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)

            return x

    def conv_layer(self, inputs, params, training):
        """
        定义卷积层，带有batch_normalization，relu
        :param inputs: 输入数据维度为 4-D tensor: [batch_size, width, height, channels]
                       or [batch_size, height, width, channels]
        :param params: 卷积层参数,[filters, kernel_size, strides, padding, batch_norm, relu, name]. [list]
        :param training: 参数是否可以训练. [boolean]
        :return:
        """
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)
        gamma_initializer = tf.random_normal_initializer(1, 0.02)

        # conv
        outputs = tf.layers.conv2d(inputs, params[0], params[1], strides=params[2],
                                   padding=params[3],
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name=params[6])

        # batch_norm
        if params[4]:
            outputs = self.norm_layer(outputs, training, name=params[6] + '/batch_norm')
            # outputs = tf.layers.batch_normalization(inputs,
            #                                         axis=3,
            #                                         epsilon=1e-5,
            #                                         momentum=0.1,
            #                                         training=training,
            #                                         gamma_initializer=gamma_initializer,
            #                                         name=params[6] + '/batch_norm')

        # relu
        if params[5]:
            outputs = tf.nn.relu(outputs, name=params[6] + '/relu')

        return outputs

    def block_resnet_others(self, inputs, layer_params, relu, training, name):
        """
        定义ResNet_block
        :param inputs: 输入. [tensor]
        :param layer_params: 卷积层参数. [list]
        :param relu: 是否使用relu激活函数. [boolean]
        :param training: 参数是否可以训练. [boolean]
        :param name: layer name. [str]
        :return:
        """
        with tf.variable_scope(name):
            short_cut = tf.identity(inputs)

            for item in layer_params:
                inputs = self.conv_layer(inputs, item, training)

            outputs = tf.add(inputs, short_cut, name='add')
            if relu:
                outputs = tf.nn.relu(outputs, 'last_relu')
        return outputs

    def conv_feat_layers(self, inputs, width, training):
        """
        cptn结构中的卷积层部分，用来提取feature_map.
        :param inputs: 输入的图像. [placeholder]
        :param width: 图像宽度. [placeholder]
        :param training:是否可训练. [boolean]
        :return:
        """
        # 卷积层各层的参数信息
        layer_params = [[64, (3, 3), (1, 1), 'same', True, True, 'conv1'],
                        [128, (3, 3), (1, 1), 'same', True, True, 'conv2'],
                        [128, (2, 2), (2, 2), 'valid', True, True, 'pool1'],
                        [128, (3, 3), (1, 1), 'same', True, True, 'conv3'],
                        [256, (3, 3), (1, 1), 'same', True, True, 'conv4'],
                        [256, (2, 2), (2, 2), 'valid', True, True, 'pool2'],
                        [256, (3, 3), (1, 1), 'same', True, True, 'conv5'],
                        [512, (3, 3), (1, 1), 'same', True, True, 'conv6'],
                        [512, (3, 2), (3, 2), 'valid', True, True, 'pool3'],
                        [512, (3, 1), (1, 1), 'valid', True, True, 'conv_feat']]

        resnet_params = [[[128, 3, (1, 1), 'same', True, True, 'conv1'],
                          [128, 3, (1, 1), 'same', True, False, 'conv2']],
                         [[256, 3, (1, 1), 'same', True, True, 'conv1'],
                          [256, 3, (1, 1), 'same', True, False, 'conv2']],
                         [[512, 3, (1, 1), 'same', True, True, 'conv1'],
                          [512, 3, (1, 1), 'same', True, False, 'conv2']]]

        # 构建卷积层
        with tf.variable_scope("conv_comm"):
            inputs = self.conv_layer(inputs, layer_params[0], training)
            inputs = self.conv_layer(inputs, layer_params[1], training)
            inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], name='padd1')
            inputs = tf.layers.max_pooling2d(inputs, (2, 2), (2, 2), 'valid', 'channels_last', 'pool1')

            inputs = self.block_resnet_others(inputs, resnet_params[0], True, training, 'res1')

            inputs = self.conv_layer(inputs, layer_params[3], training)
            inputs = self.conv_layer(inputs, layer_params[4], training)
            inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], name='padd2')
            inputs = tf.layers.max_pooling2d(inputs, (2, 2), (2, 2), 'valid', 'channels_last', 'pool2')

            inputs = self.block_resnet_others(inputs, resnet_params[1], True, training, 'res2')

            inputs = self.conv_layer(inputs, layer_params[6], training)
            inputs = self.conv_layer(inputs, layer_params[7], training)
            inputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 1], [0, 0]], name='padd3')
            inputs = tf.layers.max_pooling2d(inputs, (3, 2), (3, 2), 'valid', 'channels_last', 'pool3')

            inputs = self.block_resnet_others(inputs, resnet_params[2], True, training, 'res3')

            conv_feat = self.conv_layer(inputs, layer_params[9], training)
            feat_size = tf.shape(conv_feat)

        # 计算每个feature_map每一行的序列长度，每一行即一个序列
        two = tf.constant(2, dtype=tf.float32, name='two')
        w = tf.cast(width, tf.float32)
        for i in range(3):
            w = tf.div(w, two)
            w = tf.ceil(w)

        # 复制height倍，并转化为向量
        w = tf.cast(w, tf.int32)
        w = tf.tile(w, [feat_size[1]])
        sequence_length = tf.reshape(w, [-1], name='seq_len')  # [batch,height]

        return conv_feat, sequence_length

    def rnn_detect_layers(self, conv_feat, sequence_length, num_anchors):
        """
        Bi_LSTM + 全连接层.
        :param conv_feat: 卷积层提取到的feature map. [tensor]
        :param sequence_length: 每一行序列的长度列表，向量长度为conv_feat的高. [tensor]
        :param num_anchors: anchor的个数
        :return:
        """
        # 将feature map进行降维，因为batch_size设置为1，所以这里直接去掉batch那一维
        conv_feat = tf.squeeze(conv_feat, axis=0)
        conv_feat = tf.transpose(conv_feat, [1, 0, 2])

        # Bi_LSTM层
        en_lstm1 = tf.contrib.rnn.LSTMCell(self.rnn_size)
        en_lstm1 = tf.contrib.rnn.DropoutWrapper(en_lstm1, output_keep_prob=self.keep_prob)
        en_lstm2 = tf.contrib.rnn.LSTMCell(self.rnn_size)
        en_lstm2 = tf.contrib.rnn.DropoutWrapper(en_lstm2, output_keep_prob=self.keep_prob)
        # encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([en_lstm1])
        # encoder_cell_bw = tf.contrib.rnn.MultiRNNCell([en_lstm2])
        bi_encoder_outputs, _ = tf.nn.bidirectional_dynamic_rnn(en_lstm1,
                                                                en_lstm2,
                                                                conv_feat,
                                                                sequence_length=sequence_length,
                                                                time_major=True,
                                                                dtype=tf.float32)  # 2 * batch_size * seq_len * hidden_dim
        conv_feat = tf.concat(bi_encoder_outputs, 2)

        # 全连接层
        weight_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)

        rnn_feat = tf.layers.dense(conv_feat, self.fc_size,
                                   activation=tf.nn.relu,
                                   kernel_initializer=weight_initializer,
                                   bias_initializer=bias_initializer,
                                   name='rnn_feat')

        # 输出层，总共三个分支
        rnn_cls = tf.layers.dense(rnn_feat, num_anchors * 2,
                                  activation=tf.nn.sigmoid,
                                  kernel_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  name='text_cls')

        rnn_ver = tf.layers.dense(rnn_feat, num_anchors * 2,
                                  activation=tf.nn.tanh,
                                  kernel_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  name='text_ver')

        rnn_hor = tf.layers.dense(rnn_feat, num_anchors * 2,
                                  activation=tf.nn.tanh,
                                  kernel_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  name='text_hor')

        rnn_cls = tf.transpose(rnn_cls, perm=[1, 0, 2], name='rnn_cls')
        rnn_ver = tf.transpose(rnn_ver, perm=[1, 0, 2], name='rnn_ver')
        rnn_hor = tf.transpose(rnn_hor, perm=[1, 0, 2], name='rnn_hor')

        return rnn_cls, rnn_ver, rnn_hor

    def detect_loss(self, rnn_cls, rnn_ver, rnn_hor, target_cls, target_ver, target_hor):
        """
        模型损失函数.
        :param rnn_cls:预测得到的cls，即分类概率.
        :param rnn_ver:预测得到的ver,anchor的y坐标中心.
        :param rnn_hor:预测得到的hor,anchor的x坐标.
        :param target_cls:真实的cls
        :param target_ver:真实的ver
        :param target_hor:真实的hor
        :return:
        """
        # 计算正例和负例对应的rnn_cls
        rnn_cls_posi = rnn_cls * target_cls
        rnn_cls_neg = rnn_cls - rnn_cls_posi

        # 计算类别的平方损失
        pow_posi = tf.square(rnn_cls_posi - target_cls)
        pow_neg = tf.square(rnn_cls_neg)

        # 对损失进行加权
        mod_posi = tf.pow(pow_posi / 0.24, 5)  # 0.3, 0.2,     0.5,0.4
        mod_neg = tf.pow(pow_neg / 0.24, 5)  # 0.7, 0.6,
        mod_con = tf.pow(0.25 / 0.2, 5)

        # 统计正例和负例的个数
        num_posi = tf.reduce_sum(target_cls) / 2 + 1
        num_neg = tf.reduce_sum(target_cls + 1) / 2 - num_posi * 2 + 1

        # 计算正例和负例的损失值
        loss_cls_posi = tf.reduce_sum(pow_posi * mod_posi) / 2
        loss_cls_neg = tf.reduce_sum(pow_neg * mod_neg) / 2

        # 将正例和负例的损失分别计算平均值，最终加和，
        # 因为同一张图像会出现较多负例，所以这样要比两者加和后再计算平均好一点
        loss_cls = loss_cls_posi / num_posi + loss_cls_neg / num_neg
        print('loss_cls:%s' % str(loss_cls))

        # 计算正例的rnn_ver和rnn_hor
        rnn_ver_posi = rnn_ver * target_cls
        rnn_hor_posi = rnn_hor * target_cls

        # 计算负例的rnn_ver和rnn_hor
        rnn_ver_neg = rnn_ver - rnn_ver_posi
        rnn_hor_neg = rnn_hor - rnn_hor_posi

        # 计算正例的ver和hor平方损失
        pow_ver_posi = tf.square(rnn_ver_posi - target_ver)
        pow_hor_posi = tf.square(rnn_hor_posi - target_hor)

        # 计算负例的ver和hor的平方损失
        pow_ver_neg = tf.square(rnn_ver_neg)
        pow_hor_neg = tf.square(rnn_hor_neg)

        # 对正例的平方损失进行加权并计算平均，这里有点类似focal loss的思想
        loss_ver_posi = tf.reduce_sum(pow_ver_posi * mod_con) / num_posi
        loss_hor_posi = tf.reduce_sum(pow_hor_posi * mod_con) / num_posi

        # 对负例的平方损失进行加权并计算平均
        loss_ver_neg = tf.reduce_sum(pow_ver_neg * mod_neg) / num_neg
        loss_hor_neg = tf.reduce_sum(pow_hor_neg * mod_neg) / num_neg

        # 对正负例的ver和hor损失进行加总
        loss_ver = loss_ver_posi + loss_ver_neg
        loss_hor = loss_hor_posi + loss_hor_neg

        loss = tf.add(loss_cls, loss_ver + 2 * loss_hor, name='loss')

        return loss

