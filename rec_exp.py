import tensorflow as tf
# import networks._tf as _tf
# from networks.ops.gpu_ops import SEPCONV_MODULE
from func import *
# from pretrained import *
import functions

def get_network_pp(x, state_enc, state_dec, state_feat, motion_flag='flow_mc', in_norm=0):

    def parametric_relu(_x, name='alpha'):
        alphas = tf.get_variable(name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.25),
                                 dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5

        return pos + neg

    def one_step_rnn(tensor, state, num_filters=128, kernel=3, act=parametric_relu):

        tensor = tf.expand_dims(tensor, axis=1)
        # print(tensor.shape.as_list()[2:4])
        cell = ConvLSTMCell(shape=tensor.shape.as_list()[2:4], activation=act,
                            filters=num_filters, kernel=[kernel, kernel])

        tensor, state = tf.nn.dynamic_rnn(cell, tensor, initial_state=state, dtype=tensor.dtype)
        tensor = tf.squeeze(tensor, axis=1)

        return tensor, state

    def sepconv(tensor, kh, kv):

        t_shape = tf.shape(tensor)

        image_patches = tf.reshape(tf.image.extract_image_patches(
            tensor, ksizes=[1, 51, 51, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME'),
            (t_shape[0], t_shape[1], t_shape[2], 51, 51, t_shape[3]))

        frame = tf.reduce_sum(tf.reduce_sum(image_patches * tf.expand_dims(tf.expand_dims(kh, -2), -1)
                                            * tf.expand_dims(tf.expand_dims(kv, -1), -1), axis=-2), axis=-2)

        return frame

    def layers(tensor, down=False, up=False, filters=64, layer_num=3):
        if down:
            tensor = tf.layers.conv2d(tensor, filters, 3, strides=(2, 2), activation=parametric_relu, padding='same')
        if up:
            tensor = tf.image.resize_bilinear(tensor, [2 * tf.shape(tensor)[1], 2 * tf.shape(tensor)[2]])
        for i in range(layer_num):
            tensor = tf.layers.conv2d(tensor, filters, 3, activation=parametric_relu, padding='same')

        return tensor

    def resblock(tensor, filters, num=2):

        for i in range(num):

            l1 = tf.layers.conv2d(inputs=tensor, filters=filters, kernel_size=3, strides=1, activation=parametric_relu, padding='same')
            l2 = tf.layers.conv2d(inputs=l1, filters=filters, kernel_size=3, strides=1, activation=parametric_relu, padding='same')
            tensor += l2

        return tensor

    def subnet(tensor, out_filter=51):
        tensor = tf.image.resize_bilinear(tensor, [2 * tf.shape(tensor)[1], 2 * tf.shape(tensor)[2]])
        tensor = tf.layers.conv2d(tensor, 64, 3, activation=parametric_relu, padding='same')
        tensor = tf.layers.conv2d(tensor, out_filter, 3, padding='same')

        return tensor

    with tf.variable_scope('unet', None, [x, state_enc, state_dec, state_feat], reuse=tf.AUTO_REUSE):

        with tf.variable_scope('encoder', None, [x, state_enc, state_feat]):
            with tf.variable_scope('downscale_1', None, [x]):

                if in_norm == 1:

                    if motion_flag == 'flow':

                        x_in, flow_in = tf.split(x, [6, 4], axis=-1)
                        x_in = input_norm(x)
                        # print('input_norm')
                        x_in = tf.concat([x_in, flow_in], axis=-1)

                    else:
                        x_in = input_norm(x)
                        # print('input_norm')
                else:
                    x_in = x
                    # print('w/o input_norm')

                pool1 = layers(x_in, down=False, up=False, filters=32, layer_num=3)

            with tf.variable_scope('downscale_2', None, [pool1]):
                pool2 = layers(pool1, down=True, up=False, filters=64, layer_num=1)

            with tf.variable_scope('downscale_3', None, [pool2]):
                pool3 = layers(pool2, down=True, up=False, filters=128, layer_num=1)

            with tf.variable_scope('rec_enc', None, [pool3, state_enc]):
                pool3_rec, state_enc = one_step_rnn(pool3, state_enc)

            with tf.variable_scope('downscale_4', None, [pool3_rec]):
                pool4 = layers(pool3_rec, down=True, up=False, filters=256, layer_num=1)

            with tf.variable_scope('downscale_5', None, [pool4, state_feat]):
                pool5 = layers(pool4, down=True, up=False, filters=512, layer_num=1)
                pool5_rec, state_feat = one_step_rnn(pool5, state_feat, num_filters=512)
                pool5 = layers(pool5_rec, down=False, up=False, filters=512, layer_num=1)

        with tf.variable_scope('decoder', None, [pool5, pool4, pool3, pool2, pool1, state_dec]):

            with tf.variable_scope('upscale_4', None, [pool5, pool4]):
                up4 = layers(pool5, down=False, up=True, filters=256, layer_num=2)
                up4 += resblock(pool4, filters=256, num=1)

            with tf.variable_scope('upscale_3', None, [up4, pool3]):
                up3 = layers(up4, down=False, up=True, filters=128, layer_num=2)
                up3 += resblock(pool3, filters=128, num=1)

            with tf.variable_scope('rec_dec', None, [up3, state_dec]):
                up3_rec, state_dec = one_step_rnn(up3, state_dec)

            with tf.variable_scope('upscale_2', None, [up3_rec, pool2]):
                up2 = layers(up3_rec, down=False, up=True, filters=64, layer_num=2)
                up2 += resblock(pool2, filters=64, num=1)

        if motion_flag == 'sepconv':

            with tf.variable_scope('sepconv', None, [up2, x]):

                with tf.variable_scope('frame_1', None, [up2]):
                    kv_1 = subnet(up2)
                    kh_1 = subnet(up2)

                with tf.variable_scope('frame_2', None, [up2]):
                    kv_2 = subnet(up2)
                    kh_2 = subnet(up2)

                frame_1 = sepconv(x[:, :, :, :3], kv_1, kh_1)
                frame_2 = sepconv(x[:, :, :, 3:], kv_2, kh_2)

                output = frame_1 + frame_2

        else:

            with tf.variable_scope('flow', None, [up2, x]):

                with tf.variable_scope('frame_1', None, [up2]):
                    flow_mask_1 = subnet(up2, out_filter=3)
                    flow_1, mask_1 = tf.split(flow_mask_1, [2, 1], axis=-1)
                with tf.variable_scope('frame_2', None, [up2]):
                    flow_mask_2 = subnet(up2, out_filter=3)
                    flow_2, mask_2 = tf.split(flow_mask_2, [2, 1], axis=-1)

                frame_1 = mask_1 * tf.contrib.image.dense_image_warp(x[:, :, :, 0:3], flow_1)
                frame_2 = mask_2 * tf.contrib.image.dense_image_warp(x[:, :, :, 3:6], flow_2)

                if motion_flag == 'flow':
                    output = frame_1 + frame_2
                else:
                    with tf.variable_scope('refine', None, [x, flow_1, flow_2, mask_1, mask_2, frame_1, frame_2]):

                        input_to_refine = tf.concat([x[:, :, :, 0:3], x[:, :, :, 3:6],
                                                     flow_1, flow_2, mask_1, mask_2,
                                                     frame_1, frame_2], axis=-1)

                        output = frame_1 + frame_2 + functions.MC_RLVC(input_to_refine)

    if motion_flag == 'sepconv':

        return output, state_enc, state_dec, state_feat

    else:

        return output, state_enc, state_dec, state_feat, tf.concat([flow_1, flow_2], axis=-1)



