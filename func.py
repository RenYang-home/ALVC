import tensorflow as tf
import numpy as np

# config = tf.ConfigProto(allow_soft_placement=True)
# sess = tf.Session(config=config)
#
# batch_size = 3
# Height = 128
# Width = 128
# Channel = 3
#
# def read_png(path):
#
#     image_group = []
#
#     index = np.random.randint(1, 6)
#     # index = 2
#
#     for i in range(index, index + 3):
#
#         string = tf.read_file(path + '/im' + str(index) + '.png')
#         image = tf.image.decode_image(string, channels=3)
#         image = tf.cast(image, tf.float32)
#         image /= 255
#
#         image_group.append(image)
#
#     return tf.stack(image_group, axis=0)
#
# # with tf.device("/cpu:0"):
# train_files = np.load('/scratch_net/maja_second/compression/models/folder_vimeo.npy').tolist()
#
# train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
# train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
# train_dataset = train_dataset.map(read_png,
#                                   num_parallel_calls=16)
# train_dataset = train_dataset.map(lambda x: tf.random_crop(x, (3, Height, Width, 3)),
#                                   num_parallel_calls=16)
# train_dataset = train_dataset.batch(batch_size)
# train_dataset = train_dataset.prefetch(32)
#
# data_tensor = train_dataset.make_one_shot_iterator().get_next()
# data_tensor = tf.ensure_shape(data_tensor, (batch_size, 3, Height, Width, 3))

def channel_norm(R):

    # R_mean, R_var = tf.nn.moments(R, axes=[1, 2, 3])
    # R_mean = R_mean[:, tf.newaxis, tf.newaxis, tf.newaxis]
    # R_var = R_var[:, tf.newaxis, tf.newaxis, tf.newaxis]
    # R_dev = tf.sqrt(R_var)

    R_mean = tf.reduce_mean(R, axis=[1, 2, 3])
    R_mean = R_mean[:, tf.newaxis, tf.newaxis, tf.newaxis]
    R_dev = tf.sqrt(tf.reduce_mean(tf.square(R - R_mean), axis=[1, 2, 3]))
    R_dev = R_dev[:, tf.newaxis, tf.newaxis, tf.newaxis]

    R_norm = (R - R_mean) / (R_dev + 0.0000001)
    R1, R2 = tf.split(R_norm, 2, axis=-1)

    return R1, R2, R_mean, R_dev

def input_norm(x):

    R = tf.concat([x[:, :, :, 0:1], x[:, :, :, 3:4]], axis=-1)
    R1, R2, _, _ = channel_norm(R)

    G = tf.concat([x[:, :, :, 1:2], x[:, :, :, 4:5]], axis=-1)
    G1, G2, _, _ = channel_norm(G)

    B = tf.concat([x[:, :, :, 2:3], x[:, :, :, 5:6]], axis=-1)
    B1, B2, _, _ = channel_norm(B)

    x_norm = tf.concat([R1, G1, B1, R2, G2, B2], axis=-1)

    return x_norm

def input_norm_2(x):

    x_1, x_2 = tf.split(x, 2, axis=-1)
    x_new = tf.stack([x_1, x_2], axis=1)

    x_mean = tf.reduce_mean(x_new, axis=[1, 2, 3])
    x_mean = x_mean[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
    x_dev = tf.sqrt(tf.reduce_mean((x_new - x_mean) ** 2, axis=[1, 2, 3]))
    x_dev = x_dev[:, tf.newaxis, tf.newaxis, tf.newaxis, :]

    x_norm = (x_new - x_mean) / (x_dev + 0.0000001)
    x_1_norm, x_2_norm = tf.split(x_norm, 2, axis=1)

    x_1_norm = tf.squeeze(x_1_norm)
    x_2_norm = tf.squeeze(x_2_norm)

    return tf.concat([x_1_norm, x_2_norm], axis=-1)


def input_norm_np(in_frames, batch):

    out_frames = np.copy(in_frames)

    for b in range(batch):
        for ch in range(3):
            xx = np.concatenate([in_frames[b, :, :, ch:ch + 1], in_frames[b, :, :, ch + 3:ch + 4]], axis=-1)

            R_m_np = np.mean(xx)
            R_dev_np = np.std(xx)

            xx = (xx - R_m_np) / R_dev_np

            xx_1, xx_2 = np.split(xx, 2, axis=-1)

            out_frames[b, :, :, ch:ch + 1] = xx_1
            out_frames[b, :, :, ch + 3:ch + 4] = xx_2

    return out_frames

# [frame_in1, frame_out_gt, frame_in2] = tf.split(data_tensor, 3, axis=1)
# input_frames = tf.squeeze(tf.concat([frame_in1, frame_in2], axis=-1))
# x_output = input_norm(input_frames)
# x_output_2 = input_norm_2(input_frames)
#
# x_o, x_o_2, in_frames = sess.run([x_output, x_output_2, input_frames])
#
# xx = input_norm_np(in_frames, batch_size)
#
# mse = np.mean(np.square(x_o - x_o_2))
# psnr = 10 * np.log10(1 / mse)
# print(mse, psnr)
#
# mse = np.mean(np.square(x_o - xx))
# psnr = 10 * np.log10(1 / mse)
# print(mse, psnr)
#
# mse = np.mean(np.square(xx - x_o_2))
# psnr = 10 * np.log10(1 / mse)
# print(mse, psnr)



class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.
  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=False, peephole=False, data_format='channels_last', reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    c, h = state

    x = tf.concat([x, h], axis=self._feature_axis)
    n = x.shape[-1].value
    m = 4 * self._filters if self._filters > 1 else 4
    W = tf.get_variable('kernel', self._kernel + [n, m])
    y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
    if not self._normalize:
      y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tf.get_variable('W_ci', c.shape[1:]) * c
      f += tf.get_variable('W_cf', c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      i = tf.contrib.layers.layer_norm(i)
      f = tf.contrib.layers.layer_norm(f)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tf.get_variable('W_co', c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state

