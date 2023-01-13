import argparse
import os
import numpy as np
import helper
from functions_inter import *
from scipy import misc
import inv_flow
import mc_func

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", default='BasketballPass')
parser.add_argument("--frame", type=int, default=14)
parser.add_argument("--f_P", type=int, default=5)
parser.add_argument("--inter", type=int, default=2)
parser.add_argument("--b_P", type=int, default=5)
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--metric", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--VTM", type=int, default=1, choices=[0, 1])
parser.add_argument("--python_path", default='python')
parser.add_argument("--l", type=int, default=512, choices=[256, 512, 1024, 2048])
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])
args = parser.parse_args()

path_root = './'
path_raw = args.path + '/'

# Settings
I_level, Height, Width, batch_size, Channel, \
activation, GOP_size, GOP_num, \
path, path_com, path_bin, path_lat = helper.configure(args, path_root=path_root, path_raw=path_raw)


# Placeholder
data_tensor = tf.placeholder(tf.float32, [batch_size, 5, Height, Width, Channel])
inter_num = tf.placeholder(tf.float32, [])

[frame_left, frame_0, frame_t, frame_1, frame_right] = tf.unstack(data_tensor, axis=1)

def q_flow(flow1, flow2, t):

    T = inter_num + 1

    a = (2 * T * flow1 + 2 * flow2)/(T + T ** 2)
    v0 = (-(T ** 2) * flow1 + flow2)/(T + T ** 2)

    return 0.5 * a * (t ** 2) + v0 * t

def parametric_relu(_x, name='alpha'):
    alphas = tf.get_variable(name, _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.25),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg

with tf.variable_scope("flow_motion_short", reuse=tf.AUTO_REUSE):
    flow_0left, _, _, _, _, _ = inv_flow.optical_flow(frame_left, frame_0, batch_size, Height, Width)
with tf.variable_scope("flow_motion_long", reuse=tf.AUTO_REUSE):
    flow_01, _, _, _, _, _ = inv_flow.optical_flow(frame_1, frame_0, batch_size, Height, Width)
with tf.variable_scope("flow_motion_short", reuse=tf.AUTO_REUSE):
    flow_1right, _, _, _, _, _ = inv_flow.optical_flow(frame_right, frame_1, batch_size, Height, Width)
with tf.variable_scope("flow_motion_long", reuse=tf.AUTO_REUSE):
    flow_10, _, _, _, _, _ = inv_flow.optical_flow(frame_0, frame_1, batch_size, Height, Width)

flow_0t = q_flow(flow_0left, flow_01, t=1)
flow_1t = q_flow(flow_1right, flow_10, t=inter_num)

flow_t0 = inv_flow.reverse_flow(flow_0t, Height, Width)
flow_t1 = inv_flow.reverse_flow(flow_1t, Height, Width)

t_warp0 = tf.contrib.image.dense_image_warp(frame_0, flow_t0)
t_warp1 = tf.contrib.image.dense_image_warp(frame_1, flow_t1)

with tf.variable_scope('flow_refine'):
    refine_input = tf.concat([frame_0, frame_1, flow_01, flow_10, flow_t0, flow_t1, t_warp0, t_warp1], axis=-1)
    refine_output, feature = mc_func.refine_net(refine_input, out_channel=8)

refine_1, refine_2, refine_3, refine_4 = tf.split(refine_output, [2, 2, 2, 2], axis=-1)

flow_t0_refine = tf.contrib.image.dense_image_warp(flow_t0, 10 * tf.tanh(refine_1)) + refine_2
flow_t1_refine = tf.contrib.image.dense_image_warp(flow_t1, 10 * tf.tanh(refine_3)) + refine_4

frame_t_warp0 = tf.contrib.image.dense_image_warp(frame_0, flow_t0_refine)
frame_t_warp1 = tf.contrib.image.dense_image_warp(frame_1, flow_t1_refine)

with tf.variable_scope("masknet"):
    mask_input = tf.concat([frame_t_warp0, frame_t_warp1, feature], axis=-1)
    tensor_mask = tf.layers.conv2d(inputs=mask_input, filters=32, kernel_size=5, strides=1, activation=parametric_relu, padding='same')
    tensor_mask = tf.layers.conv2d(inputs=tensor_mask, filters=16, kernel_size=3, strides=1, activation=parametric_relu, padding='same')
    tensor_mask = tf.sigmoid(tf.layers.conv2d(inputs=tensor_mask, filters=2, kernel_size=3, strides=1, padding='same'))

frame_t_warp = tf.div_no_nan((frame_t_warp0 * tensor_mask[:, :, :, 0:1] * inter_num + frame_t_warp1 * tensor_mask[:, :, :, 1:2]),
                             (tensor_mask[:, :, :, 0:1] * inter_num + tensor_mask[:, :, :, 1:2]))

with tf.variable_scope('post'):

    input_to_post = tf.concat([frame_0, frame_1, frame_t_warp, tensor_mask, frame_t_warp0, frame_t_warp1, flow_t0_refine, flow_t1_refine], axis=-1)
    output = frame_t_warp + mc_func.MC_RLVC(input_to_post)

entropy_mv = tfc.EntropyBottleneck()
entropy_res = tfc.EntropyBottleneck()

frame_t_com, mse_loss, psnr_loss, bpp_loss, flow_lat, res_lat = \
    DVC_compress(output, frame_t, entropy_mv, entropy_res, batch_size, Height, Width, args, training=False)

psnr_value = np.load(path_bin + 'quality.npy')
bpp_value = np.load(path_bin + 'bpp.npy')

sess.run(tf.global_variables_initializer())
all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
saver = tf.train.Saver(max_to_keep=None, var_list=all_var)
model_path = './model/Interpolation/lambda_' + str(args.l) + '_inter_' + str(args.inter) + '/'
saver.restore(sess, save_path=model_path + 'model.ckpt-200000')

# encode GOPs
for g in range(GOP_num):

    F_left = misc.imread(path_com + 'f' + str(g * GOP_size + args.f_P).zfill(3) + '.png').astype(float)
    F_0 = misc.imread(path_com + 'f' + str(g * GOP_size + args.f_P + 1).zfill(3) + '.png').astype(float)
    F_1 = misc.imread(path_com + 'f' + str(g * GOP_size + args.f_P + args.inter + 2).zfill(3) + '.png').astype(float)
    F_right = misc.imread(path_com + 'f' + str(g * GOP_size + args.f_P + args.inter + 3).zfill(3) + '.png').astype(float)

    G_t = misc.imread(path_raw + 'f' + str(g * GOP_size + args.f_P + 2).zfill(3) + '.png').astype(float)

    input_data = np.stack([F_left, F_0, G_t, F_1, F_right], axis=0)
    input_data = np.expand_dims(input_data/255.0, axis=0)

    psnr, bpp, F_t, mv_latent, res_latent \
        = sess.run([psnr_loss, bpp_loss, frame_t_com, flow_lat, res_lat],
                   feed_dict={data_tensor:input_data, inter_num:args.inter})

    # with open(path_bin + '/f' + str(g * GOP_size + args.f_P + 2).zfill(3) + '.bin', "wb") as ff:
    #     ff.write(np.array(len(string_MV), dtype=np.uint16).tobytes())
    #     ff.write(string_MV)
    #     ff.write(string_Res)

    F_t = np.clip(F_t, 0, 1)
    F_t = np.uint8(F_t * 255.0)

    psnr_value[g * GOP_size + args.f_P + 1] = psnr
    bpp_value[g * GOP_size + args.f_P + 1] = bpp

    print('Frame', g * GOP_size + args.f_P + 2, args.metric + ' =', psnr)

    misc.imsave(path_com + 'f' + str(g * GOP_size + args.f_P + 2).zfill(3) + '.png', F_t[0])
    np.save(path_lat + 'f' + str(g * GOP_size + args.f_P + 2).zfill(3) + '_mv.npy', mv_latent)
    np.save(path_lat + 'f' + str(g * GOP_size + args.f_P + 2).zfill(3) + '_res.npy', res_latent)

sess.run(tf.global_variables_initializer())
all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
saver = tf.train.Saver(max_to_keep=None, var_list=all_var)
model_path = './model/Interpolation/lambda_' + str(args.l) + '_inter_' + str(args.inter - 1) + '/'
saver.restore(sess, save_path=model_path + 'model.ckpt-200000')

# encode GOPs
for g in range(GOP_num):

    # print(GOP_size, g * GOP_size + args.f_P + args.inter + 3)
    F_left = misc.imread(path_com + 'f' + str(g * GOP_size + args.f_P + 1).zfill(3) + '.png').astype(float)
    F_0 = misc.imread(path_com + 'f' + str(g * GOP_size + args.f_P + 2).zfill(3) + '.png').astype(float)
    F_1 = misc.imread(path_com + 'f' + str(g * GOP_size + args.f_P + args.inter + 2).zfill(3) + '.png').astype(float)
    F_right = misc.imread(path_com + 'f' + str(g * GOP_size + args.f_P + args.inter + 3).zfill(3) + '.png').astype(float)

    G_t = misc.imread(path_raw + 'f' + str(g * GOP_size + args.f_P + args.inter + 1).zfill(3) + '.png').astype(float)

    input_data = np.stack([F_left, F_0, G_t, F_1, F_right], axis=0)
    input_data = np.expand_dims(input_data / 255.0, axis=0)

    psnr, bpp, F_t, mv_latent, res_latent\
        = sess.run([psnr_loss, bpp_loss, frame_t_com, flow_lat, res_lat],
                   feed_dict={data_tensor: input_data, inter_num:args.inter - 1})

    # with open(path_bin + '/f' + str(g * GOP_size + args.f_P + 1).zfill(3) + '.bin', "wb") as ff:
    #     ff.write(np.array(len(string_MV), dtype=np.uint16).tobytes())
    #     ff.write(string_MV)
    #     ff.write(string_Res)

    F_t = np.clip(F_t, 0, 1)
    F_t = np.uint8(F_t * 255.0)

    psnr_value[g * GOP_size + args.f_P + args.inter] = psnr
    bpp_value[g * GOP_size + args.f_P + args.inter] = bpp

    print('Frame', g * GOP_size + args.f_P + args.inter + 1, args.metric + ' =', psnr)

    misc.imsave(path_com + 'f' + str(g * GOP_size + args.f_P + args.inter + 1).zfill(3) + '.png', F_t[0])
    np.save(path_lat + 'f' + str(g * GOP_size + args.f_P + args.inter + 1).zfill(3) + '_mv.npy', mv_latent)
    np.save(path_lat + 'f' + str(g * GOP_size + args.f_P + args.inter + 1).zfill(3) + '_res.npy', res_latent)

print('Average: ' + args.path, np.average(psnr_value), np.average(bpp_value))

np.save(path_bin + '/quality.npy', psnr_value)
np.save(path_bin + '/bpp.npy', bpp_value)



