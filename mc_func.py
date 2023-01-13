import tensorflow as tf
import numpy as np

def resblock(input, IC, OC, name, reuse=tf.AUTO_REUSE):

    l1 = tf.nn.relu(input, name=name + 'relu1')

    l1 = tf.layers.conv2d(inputs=l1, filters=np.minimum(IC, OC), kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'l1', reuse=reuse)

    l2 = tf.nn.relu(l1, name='relu2')

    l2 = tf.layers.conv2d(inputs=l2, filters=OC, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'l2', reuse=reuse)

    if IC != OC:
        input = tf.layers.conv2d(inputs=input, filters=OC, kernel_size=1, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name=name + 'map', reuse=reuse)

    return input + l2

def resblock_init(input, IC, OC, name, reuse=tf.AUTO_REUSE):

    l1 = tf.nn.relu(input, name=name + 'relu1')

    l1 = tf.layers.conv2d(inputs=l1, filters=np.minimum(IC, OC), kernel_size=3, strides=1, padding='same', name=name + 'l1', reuse=reuse)

    l2 = tf.nn.relu(l1, name='relu2')

    l2 = tf.layers.conv2d(inputs=l2, filters=OC, kernel_size=3, strides=1, padding='same', name=name + 'l2', reuse=reuse)

    if IC != OC:
        input = tf.layers.conv2d(inputs=input, filters=OC, kernel_size=1, strides=1, padding='same', name=name + 'map', reuse=reuse)

    return input + l2

def MC_RLVC(input, reuse=tf.AUTO_REUSE):

    m1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1', reuse=reuse)

    m2 = resblock(m1, 64, 64, name='mc2', reuse=reuse)

    m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')

    m4 = resblock(m3, 64, 64, name='mc4', reuse=reuse)

    m5 = tf.layers.average_pooling2d(m4, pool_size=2, strides=2, padding='same')

    m6 = resblock(m5, 64, 64, name='mc6', reuse=reuse)

    m7 = resblock(m6, 64, 64, name='mc7', reuse=reuse)

    m8 = tf.image.resize_images(m7, [2 * tf.shape(m7)[1], 2 * tf.shape(m7)[2]])

    m8 = m4 + m8

    m9 = resblock(m8, 64, 64, name='mc9', reuse=reuse)

    m10 = tf.image.resize_images(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])

    m10 = m2 + m10

    m11 = resblock(m10, 64, 64, name='mc11', reuse=reuse)

    m12 = tf.layers.conv2d(inputs=m11, filters=64, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12', reuse=reuse)

    m12 = tf.nn.relu(m12, name='relu12')

    m13 = tf.layers.conv2d(inputs=m12, filters=3, kernel_size=3, strides=1, padding='same',
                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc13', reuse=reuse)

    return m13

def MC_RLVC_init(input, reuse=tf.AUTO_REUSE):

    m1 = tf.layers.conv2d(inputs=input, filters=64, kernel_size=3, strides=1, padding='same', name='mc1', reuse=reuse)

    m2 = resblock_init(m1, 64, 64, name='mc2', reuse=reuse)

    m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')

    m4 = resblock_init(m3, 64, 64, name='mc4', reuse=reuse)

    m5 = tf.layers.average_pooling2d(m4, pool_size=2, strides=2, padding='same')

    m6 = resblock_init(m5, 64, 64, name='mc6', reuse=reuse)

    m7 = resblock_init(m6, 64, 64, name='mc7', reuse=reuse)

    m8 = tf.image.resize_images(m7, [2 * tf.shape(m7)[1], 2 * tf.shape(m7)[2]])

    m8 = m4 + m8

    m9 = resblock_init(m8, 64, 64, name='mc9', reuse=reuse)

    m10 = tf.image.resize_images(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])

    m10 = m2 + m10

    m11 = resblock_init(m10, 64, 64, name='mc11', reuse=reuse)

    m12 = tf.layers.conv2d(inputs=m11, filters=64, kernel_size=3, strides=1, padding='same', name='mc12', reuse=reuse)

    m12 = tf.nn.relu(m12, name='relu12')

    m13 = tf.layers.conv2d(inputs=m12, filters=3, kernel_size=3, strides=1, padding='same', name='mc13', reuse=reuse)

    return m13

def MC_light(input, filter_num = 32, out_filter=3, name='post_light', reuse=tf.AUTO_REUSE):

    with tf.variable_scope(name, reuse=reuse):

        m1 = tf.layers.conv2d(inputs=input, filters=filter_num, kernel_size=3, strides=1, padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc1')

        m2 = resblock(m1, filter_num, filter_num, name='mc2')

        # m3 = tf.layers.average_pooling2d(m2, pool_size=2, strides=2, padding='same')
        #
        # m4 = resblock(m3, filter_num, filter_num, name='mc4')
        #
        # m9 = resblock(m4, filter_num, filter_num, name='mc9')
        #
        # m10 = tf.image.resize_images(m9, [2 * tf.shape(m9)[1], 2 * tf.shape(m9)[2]])
        #
        # m10 = m2 + m10

        m11 = resblock(m2, filter_num, filter_num, name='mc11') + m1

        m12 = tf.layers.conv2d(inputs=m11, filters=filter_num, kernel_size=3, strides=1, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc12')

        m12 = tf.nn.relu(m12, name='relu12')

        m13 = tf.layers.conv2d(inputs=m12, filters=out_filter, kernel_size=3, strides=1, padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True), name='mc13')

    return m13


def refine_net(x, out_channel):
    def parametric_relu(_x, name='alpha'):
        alphas = tf.get_variable(name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.25),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

    def layers(tensor, down=False, up=False, filters=64, layer_num=3, f_size=3):
        if down:
            tensor = tf.layers.average_pooling2d(tensor, 2, 2, padding='same')
        if up:
            tensor = tf.image.resize_bilinear(tensor, [2 * tf.shape(tensor)[1], 2 * tf.shape(tensor)[2]])
        for i in range(layer_num):
            tensor = tf.layers.conv2d(tensor, filters, f_size, activation=parametric_relu, padding='same')

        return tensor

    with tf.variable_scope('unet', None, [x]):

        with tf.variable_scope('encoder', None, [x]):
            with tf.variable_scope('downscale_1', None, [x]):
                pool1 = layers(x, down=False, up=False, filters=32, layer_num=2, f_size=7)

            with tf.variable_scope('downscale_2', None, [pool1]):
                pool2 = layers(pool1, down=True, up=False, filters=64, layer_num=2, f_size=5)

            with tf.variable_scope('downscale_3', None, [pool2]):
                pool3 = layers(pool2, down=True, up=False, filters=128, layer_num=2)

            with tf.variable_scope('downscale_4', None, [pool3]):
                pool4 = layers(pool3, down=True, up=False, filters=256, layer_num=2)

            with tf.variable_scope('downscale_5', None, [pool4]):
                pool5 = layers(pool4, down=True, up=False, filters=512, layer_num=3)

        with tf.variable_scope('decoder', None, [pool5, pool4, pool3, pool2, pool1]):

            with tf.variable_scope('upscale_4', None, [pool5, pool4]):
                up4 = layers(pool5, down=False, up=True, filters=256, layer_num=1)
                up4 = tf.concat([up4, pool4], axis=-1)
                up4 = tf.layers.conv2d(up4, 256, 3, activation=parametric_relu, padding='same')

            with tf.variable_scope('upscale_3', None, [up4, pool3]):
                up3 = layers(up4, down=False, up=True, filters=128, layer_num=1)
                up3 = tf.concat([up3, pool3], axis=-1)
                up3 = tf.layers.conv2d(up3, 128, 3, activation=parametric_relu, padding='same')

            with tf.variable_scope('upscale_2', None, [up3, pool2]):
                up2 = layers(up3, down=False, up=True, filters=64, layer_num=1)
                up2 = tf.concat([up2, pool2], axis=-1)
                up2 = tf.layers.conv2d(up2, 64, 3, activation=parametric_relu, padding='same')

            with tf.variable_scope('upscale_1', None, [up2, pool1]):
                up1 = layers(up2, down=False, up=True, filters=32, layer_num=1)
                up1 = tf.concat([up1, pool1], axis=-1)
                up1 = tf.layers.conv2d(up1, 32, 3, activation=parametric_relu, padding='same')

        with tf.variable_scope('output', None, [up1]):

            output = tf.layers.conv2d(up1, out_channel, 3, padding='same')

    return output, up1

