import argparse
import numpy as np
import tensorflow as tf
import os
import CNN_recurrent
import helper2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# use CPU for RPM to ensure the determinism
# config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", default='BasketballPass')
parser.add_argument("--frame", type=int, default=100)
parser.add_argument("--f_P", type=int, default=5)
parser.add_argument("--inter", type=int, default=2)
parser.add_argument("--b_P", type=int, default=5)
parser.add_argument("--mode", default='PSNR', choices=['PSNR', 'MS-SSIM'])
parser.add_argument("--l", type=int, default=512, choices=[256, 512, 1024, 2048])
parser.add_argument("--entropy_coding", type=int, default=0)
parser.add_argument("--N", type=int, default=128, choices=[128])
parser.add_argument("--M", type=int, default=128, choices=[128])
args = parser.parse_args()

# Settings
I_level, Height, Width, batch_size, Channel, \
activation, GOP_size, GOP_num, \
path, path_com, path_bin, path_lat = helper2.configure(args)

# Placeholder
prior_tensor = tf.placeholder(tf.float32, [batch_size, Height//16, Width//16, args.M]) # previous latent
latent_tensor = tf.placeholder(tf.float32, [batch_size, Height//16, Width//16, args.M]) # latent to compress

hidden_states = tf.placeholder(tf.float32, [2, batch_size, Height//16, Width//16, args.N]) # hidden states in RPM

c_prob, h_prob = tf.split(hidden_states, 2, axis=0)

# RPM network
prob_latent, c_prob_out, h_prob_out \
    = CNN_recurrent.rec_prob(prior_tensor, args.N, Height, Width, c_prob[0], h_prob[0])

# estimate bpp
bits_est, sigma, mu = CNN_recurrent.bpp_est(latent_tensor, prob_latent, args.N)

hidden_states_out = tf.stack([c_prob_out, h_prob_out], axis = 0)

# calculates bits for I frames and bottlenecks
total_bits = 0
bpp = np.zeros([args.frame])

for g in range(GOP_num + 1):

    I_index = g * GOP_size + 1

    if I_index <= args.frame:

        if os.path.exists(path_bin + 'f' + str(I_index).zfill(3) + '.bin'):
            total_bits += os.path.getsize(path_bin + 'f' + str(I_index).zfill(3) + '.bin') * 8
            bpp[I_index - 1] = os.path.getsize(path_bin + 'f' + str(I_index).zfill(3) + '.bin') * 8 /Height/Width
        else:
            bpp_I = np.load(path_bin + 'f' + str(I_index).zfill(3) + '.npy')
            total_bits += bpp_I * Height * Width
            bpp[I_index - 1] = bpp_I

        # if there exists forward P frame(s), I_index + 1 is encoded by the bottleneck
        if args.f_P > 0 and I_index + 1 <= args.frame:
            total_bits += os.path.getsize(path_bin + 'f' + str(I_index + 1).zfill(3) + '.bin') * 8
            bpp[I_index] = os.path.getsize(path_bin + 'f' + str(I_index + 1).zfill(3) + '.bin') * 8 / Height / Width

        # if there exists backward P frame(s), I_index - 1 is encoded by the bottleneck
        if args.b_P > 0 and I_index - 1 >= 1:
            total_bits += os.path.getsize(path_bin + 'f' + str(I_index - 1).zfill(3) + '.bin') * 8
            bpp[I_index - 2] = os.path.getsize(path_bin + 'f' + str(I_index - 1).zfill(3) + '.bin') * 8 / Height / Width

# start RPM

latents = ['mv', 'res'] # two kinds of latents

for lat in latents:

    # load model
    model_path = './model/RPM/RPM_' + args.mode + '_' + str(args.l) + '_' + lat
    saver = tf.train.Saver(max_to_keep=None)
    saver.restore(sess, save_path=model_path + '/model.ckpt')

    # encode GOPs
    for g in range(GOP_num):

        # forward P frames (only if more than 2 P frames exist)
        if args.f_P >= 2:
            # load first prior
            frame_index = g * GOP_size + 2
            prior_value = np.load(path_lat + '/f' + str(frame_index).zfill(3) + '_' + lat + '.npy')

            # init state
            h_state = np.zeros([2, batch_size, Height // 16, Width // 16, args.N], dtype=np.float)

            for f in range(args.f_P - 1):

                # load latent
                frame_index = g * GOP_size + f + 3
                latent_value = np.load(path_lat + '/f' + str(frame_index).zfill(3) + '_' + lat + '.npy')

                # run RPM
                bits_estimation, sigma_value, mu_value, h_state \
                    = sess.run([bits_est, sigma, mu, hidden_states_out],
                               feed_dict={prior_tensor: prior_value, latent_tensor: latent_value,
                               hidden_states: h_state})

                if args.entropy_coding:
                    bits_value = helper2.entropy_coding(frame_index, lat, path_bin, latent_value, sigma_value, mu_value)
                    total_bits += bits_value
                    # print('Frame', frame_index, lat + '_bits =', bits_value)
                else:
                    total_bits += bits_estimation
                    # print('Frame', frame_index, lat + '_bits =', bits_estimation)
                    bpp[frame_index - 1] += bits_estimation / Height / Width

                # the latent will be the prior for the next latent
                prior_value = latent_value

        # backward P frames (only if more than 2 P frames exist)
        if args.b_P >= 2:
            # load first prior
            frame_index = (g + 1) * GOP_size
            prior_value = np.load(path_lat + '/f' + str(frame_index).zfill(3) + '_' + lat + '.npy')

            # init state
            h_state = np.zeros([2, batch_size, Height // 16, Width // 16, args.N], dtype=np.float)

            for f in range(args.b_P - 1):

                # load latent
                frame_index = (g + 1) * GOP_size - f - 1
                latent_value = np.load(path_lat + '/f' + str(frame_index).zfill(3) + '_' + lat + '.npy')

                # run RPM
                bits_estimation, sigma_value, mu_value, h_state \
                    = sess.run([bits_est, sigma, mu, hidden_states_out],
                               feed_dict={prior_tensor: prior_value, latent_tensor: latent_value,
                                          hidden_states: h_state})

                if args.entropy_coding:
                    bits_value = helper2.entropy_coding(frame_index, lat, path_bin, latent_value, sigma_value, mu_value)
                    total_bits += bits_value
                    # print('Frame', frame_index, lat + '_bits =', bits_value)
                else:
                    total_bits += bits_estimation
                    # print('Frame', frame_index, lat + '_bits =', bits_estimation)
                    bpp[frame_index - 1] += bits_estimation / Height / Width

                # the latent will be the prior for the next latent
                prior_value = latent_value

    # encode rest frames (only if more than 2 P frames exist)
    rest_frame_num = args.frame - 1 - GOP_size * GOP_num

    if rest_frame_num >= 2:
        # load first prior
        frame_index = GOP_num * GOP_size + 2
        prior_value = np.load(path_lat + '/f' + str(frame_index).zfill(3) + '_' + lat + '.npy')

        # init state
        h_state = np.zeros([2, batch_size, Height // 16, Width // 16, args.N], dtype=np.float)

        for f in range(rest_frame_num - 1):

            # load latent
            frame_index = GOP_num * GOP_size + f + 3
            latent_value = np.load(path_lat + '/f' + str(frame_index).zfill(3) + '_' + lat + '.npy')

            # run RPM
            bits_estimation, sigma_value, mu_value, h_state \
                = sess.run([bits_est, sigma, mu, hidden_states_out],
                           feed_dict={prior_tensor: prior_value, latent_tensor: latent_value,
                                      hidden_states: h_state})

            if args.entropy_coding:
                bits_value = helper2.entropy_coding(frame_index, lat, path_bin, latent_value, sigma_value, mu_value)
                total_bits += bits_value
                # print('Frame', frame_index, lat + '_bits =', bits_value)
            else:
                total_bits += bits_estimation
                # print('Frame', frame_index, lat + '_bits =', bits_estimation)
                bpp[frame_index - 1] += bits_estimation / Height / Width

            # the latent will be the prior for the next latent
            prior_value = latent_value

bpp_video = total_bits/args.frame/Height/Width

os.system('rm ' + path_bin + '/*.bin')
np.save(path_bin + 'bpp.npy', bpp)























