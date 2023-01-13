import tensorflow as tf
import numpy as np

def convnet(im1_warp, im2, flow, layer):

    with tf.variable_scope("flow_cnn_" + str(layer), reuse=tf.AUTO_REUSE):

        input = tf.concat([im1_warp, im2, flow], axis=-1)

        conv1 = tf.layers.conv2d(inputs=input, filters=32, kernel_size=[7, 7], padding="same",
                                 activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[7, 7], padding="same",
                                 activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=32, kernel_size=[7, 7], padding="same",
                                 activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=16, kernel_size=[7, 7], padding="same",
                                 activation=tf.nn.relu)
        conv5 = tf.layers.conv2d(inputs=conv4, filters=2 , kernel_size=[7, 7], padding="same",
                                 activation=None)

    return conv5


def loss(flow_course, im1, im2, layer):

    flow = tf.image.resize_images(flow_course, [tf.shape(im1)[1], tf.shape(im2)[2]])
    im1_warped = tf.contrib.image.dense_image_warp(im1, flow)
    res = convnet(im1_warped, im2, flow, layer)
    flow_fine = res + flow

    im1_warped_fine = tf.contrib.image.dense_image_warp(im1, flow_fine)
    loss_layer = tf.reduce_mean(tf.squared_difference(im1_warped_fine, im2))

    return loss_layer, flow_fine


def optical_flow(im1_4, im2_4, batch, h, w):

    im1_3 = tf.layers.average_pooling2d(im1_4, pool_size=2, strides=2, padding='same')
    im1_2 = tf.layers.average_pooling2d(im1_3, pool_size=2, strides=2, padding='same')
    im1_1 = tf.layers.average_pooling2d(im1_2, pool_size=2, strides=2, padding='same')
    im1_0 = tf.layers.average_pooling2d(im1_1, pool_size=2, strides=2, padding='same')

    im2_3 = tf.layers.average_pooling2d(im2_4, pool_size=2, strides=2, padding='same')
    im2_2 = tf.layers.average_pooling2d(im2_3, pool_size=2, strides=2, padding='same')
    im2_1 = tf.layers.average_pooling2d(im2_2, pool_size=2, strides=2, padding='same')
    im2_0 = tf.layers.average_pooling2d(im2_1, pool_size=2, strides=2, padding='same')

    flow_zero = tf.zeros([batch, h//16, w//16, 2])

    loss_0, flow_0 = loss(flow_zero, im1_0, im2_0, 0)
    loss_1, flow_1 = loss(flow_0, im1_1, im2_1, 1)
    loss_2, flow_2 = loss(flow_1, im1_2, im2_2, 2)
    loss_3, flow_3 = loss(flow_2, im1_3, im2_3, 3)
    loss_4, flow_4 = loss(flow_3, im1_4, im2_4, 4)

    return flow_4, loss_0, loss_1, loss_2, loss_3, loss_4


def reverse_sample(x_shift, y_shift, h, w, weight):

    x, y = tf.meshgrid(tf.range(h), tf.range(w), indexing='ij')

    x = tf.cast(tf.expand_dims(x, -1), tf.float32)
    y = tf.cast(tf.expand_dims(y, -1), tf.float32)

    x -= x_shift
    y -= y_shift

    x = tf.clip_by_value(x, 0, h - 1)
    y = tf.clip_by_value(y, 0, w - 1)

    grid1 = tf.concat([x, y], axis=-1)
    grid1 = tf.cast(grid1, tf.int32)

    tf_zeros = tf.zeros([h, w, 1, 1], tf.int32)
    indices = tf.expand_dims(grid1, 2)
    indices = tf.concat([indices, tf_zeros], axis=-1)

    ref_x = tf.Variable(tf.zeros([h, w, 1], np.float32), trainable=False, dtype=tf.float32)
    ref_y = tf.Variable(tf.zeros([h, w, 1], np.float32), trainable=False, dtype=tf.float32)
    ref_w = tf.Variable(tf.zeros([h, w, 1], np.float32) + 1e-9, trainable=False, dtype=tf.float32)

    inv_flow_x = tf.scatter_nd_update(ref_x, indices, -x_shift * weight)
    inv_flow_y = tf.scatter_nd_update(ref_y, indices, -y_shift * weight)

    inv_flow_batch = tf.expand_dims(tf.concat([inv_flow_x, inv_flow_y], axis=-1), axis=0)

    weight_x = tf.scatter_nd_update(ref_w, indices, weight)

    weight_batch = tf.expand_dims(weight_x, axis=0)

    return inv_flow_batch, weight_batch

def reverse_flow(flow_input, h, w):

    flow_list = tf.unstack(flow_input)

    inv_flow = []

    for flow in flow_list:

        x_flow, y_flow = tf.split(flow, [1, 1], axis=-1)
        x_1 = tf.floor(x_flow)
        x_2 = x_1 + 1
        y_1 = tf.floor(y_flow)
        y_2 = y_1 + 1

        weight_1 = tf.exp(-((x_flow - x_1) ** 2 + (y_flow - y_1) ** 2))
        weight_2 = tf.exp(-((x_flow - x_1) ** 2 + (y_flow - y_2) ** 2))
        weight_3 = tf.exp(-((x_flow - x_2) ** 2 + (y_flow - y_1) ** 2))
        weight_4 = tf.exp(-((x_flow - x_2) ** 2 + (y_flow - y_2) ** 2))

        inv_flow_1, norm_1 = reverse_sample(x_1, y_1, h, w, weight_1)
        inv_flow_2, norm_2 = reverse_sample(x_1, y_2, h, w, weight_2)
        inv_flow_3, norm_3 = reverse_sample(x_2, y_1, h, w, weight_3)
        inv_flow_4, norm_4 = reverse_sample(x_2, y_2, h, w, weight_4)

        inv_flow_batch = inv_flow_1 + inv_flow_2 + inv_flow_3 + inv_flow_4
        norm_batch = norm_1 + norm_2 + norm_3 + norm_4

        inv_flow_norm = tf.divide(inv_flow_batch, norm_batch)
        inv_flow.append(inv_flow_norm)

    inv_flow = tf.concat(inv_flow, axis=0)

    return inv_flow
