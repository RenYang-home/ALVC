import tensorflow as tf
import tensorflow_compression as tfc
import sepconv_inter_enc as nn
import sepconv_inter as nn_inter
import motion
import MC_network_inter
import CNN_img
import numpy as np

def inter_short(frame_1, frame_2):

    frames = tf.concat([frame_1, frame_2], axis=-1)
    output_frame, _ = nn_inter.get_network_pp(frames, motion_flag='flow_mc')

    return output_frame

def inter_long(ref_frames):

    [frame_in0, frame_in1, frame_in2, frame_in3] = tf.split(ref_frames, 4, axis=1)

    input_pair_1 = tf.squeeze(tf.concat([frame_in0, frame_in1], axis=-1), axis=1)
    input_pair_2 = tf.squeeze(tf.concat([frame_in1, frame_in2], axis=-1), axis=1)
    input_pair_3 = tf.squeeze(tf.concat([frame_in2, frame_in3], axis=-1), axis=1)
    input_all = tf.squeeze(tf.concat([frame_in0, frame_in1, frame_in2, frame_in3], axis=-1), axis=1)

    pool5_1, skip4_1, skip3_1, skip2_1 = nn.get_network_enc(input_pair_1, 'enc_1')
    pool5_2, skip4_2, skip3_2, skip2_2 = nn.get_network_enc(input_pair_2, 'enc_2')
    pool5_3, skip4_3, skip3_3, skip2_3 = nn.get_network_enc(input_pair_3, 'enc_3')

    pool5 = nn.conv_map(pool5_1, pool5_2, pool5_3, 512, 'map_5')
    skip4 = nn.conv_map(skip4_1, skip4_2, skip4_3, 256, 'map_4')
    skip3 = nn.conv_map(skip3_1, skip3_2, skip3_3, 128, 'map_3')
    skip2 = nn.conv_map(skip2_1, skip2_2, skip2_3, 64, 'map_2')

    output_frame = nn.get_network_dec(pool5, skip4, skip3, skip2, input_all)

    return output_frame

def DVC_compress(Y0_com, Y1_raw, entropy_mv, entropy_res, batch_size, Height, Width, args, training=True):

    with tf.variable_scope("flow_motion"):

        flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)

    # Encode flow
    flow_latent = CNN_img.MV_analysis(flow_tensor, args.N, args.M)

    string_mv = entropy_mv.compress(flow_latent)
    # string_mv = tf.squeeze(string_mv, axis=0)

    flow_latent_hat, MV_likelihoods = entropy_mv(flow_latent, training=training)

    flow_hat = CNN_img.MV_synthesis(flow_latent_hat, args.N)

    # Motion Compensation
    Y1_warp = tf.contrib.image.dense_image_warp(Y0_com, flow_hat)

    MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
    Y1_MC = MC_network_inter.MC(MC_input)

    # Encode residual
    Res = Y1_raw - Y1_MC

    res_latent = CNN_img.Res_analysis(Res, num_filters=args.N, M=args.M)

    string_res = entropy_res.compress(res_latent)
    # string_res = tf.squeeze(string_res, axis=0)

    res_latent_hat, Res_likelihoods = entropy_res(res_latent, training=training)

    Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=args.N)

    # Reconstructed frame
    Y1_com = Res_hat + Y1_MC

    # Total number of bits divided by number of pixels.
    train_bpp_MV = tf.reduce_sum(tf.log(MV_likelihoods)) / (-np.log(2) * Height * Width * batch_size)
    train_bpp_Res = tf.reduce_sum(tf.log(Res_likelihoods)) / (-np.log(2) * Height * Width * batch_size)
    train_bpp = train_bpp_MV + train_bpp_Res

    # Mean squared error across pixels.
    # if args.mode == 'PSNR':
    total_mse = tf.reduce_mean(tf.squared_difference(Y1_com, Y1_raw))
    psnr = 10.0*tf.log(1.0/total_mse)/tf.log(10.0)
    # else:
    #     total_mse = 0
    #     psnr = tf.math.reduce_mean(tf.image.ssim_multiscale(tf.clip_by_value(Y1_com, 0, 1), Y1_raw, max_val=1))

    return Y1_com, total_mse, psnr, train_bpp, flow_latent_hat, res_latent_hat


