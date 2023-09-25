import tensorflow as tf
import rec_exp as nn
import numpy as np
import argparse
import imageio
# from func import *
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# config = tf.ConfigProto(allow_soft_placement=True, device_count={'GPU': 0})
config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--motion", type=str, default='flow_mc', choices=['flow', 'flow_mc', 'sepconv'])
parser.add_argument("--input_norm", type=int, default=0, choices=[0, 1])
parser.add_argument("--ens", type=int, default=0, choices=[0, 1])
parser.add_argument("--path", type=str)
parser.add_argument("--idx", type=int, default=3)
parser.add_argument("--l", type=int, default=1024, choices=[256, 512, 1024, 2048])
parser.add_argument("--dirc", type=str, default='fw', choices=['fw', 'bw'])
args = parser.parse_args()

os.makedirs(args.path + '/extra_states', exist_ok=True)

if args.dirc == 'fw':
    frame_1 = imageio.imread(args.path + 'f' + str(args.idx - 2).zfill(3) + '.png').astype(np.float32) / 255.0
    frame_2 = imageio.imread(args.path + 'f' + str(args.idx - 1).zfill(3) + '.png').astype(np.float32) / 255.0
else:
    frame_1 = imageio.imread(args.path + 'f' + str(args.idx + 2).zfill(3) + '.png').astype(np.float32) / 255.0
    frame_2 = imageio.imread(args.path + 'f' + str(args.idx + 1).zfill(3) + '.png').astype(np.float32) / 255.0

batch_size = 1
Height = frame_1.shape[0]
Width = frame_1.shape[1]

frame_1 = np.expand_dims(frame_1, axis=0)
frame_2 = np.expand_dims(frame_2, axis=0)

if not os.path.exists(args.path + '/extra_states/state_enc_1.npy'):
    state_enc_1 = np.zeros([batch_size, Height // 4, Width // 4, 128], dtype=np.float32)
    state_enc_2 = np.zeros([batch_size, Height // 4, Width // 4, 128], dtype=np.float32)
    state_dec_1 = np.zeros([batch_size, Height // 4, Width // 4, 128], dtype=np.float32)
    state_dec_2 = np.zeros([batch_size, Height // 4, Width // 4, 128], dtype=np.float32)
    state_fea_1 = np.zeros([batch_size, Height // 16, Width // 16, 512], dtype=np.float32)
    state_fea_2 = np.zeros([batch_size, Height // 16, Width // 16, 512], dtype=np.float32)

    if args.motion == 'flow' or args.motion == 'flow_mc':
        flow = np.zeros([batch_size, Height, Width, 4], dtype=np.float32)

else:

    state_enc_1 = np.load(args.path + '/extra_states/state_enc_1.npy')
    state_enc_2 = np.load(args.path + '/extra_states/state_enc_2.npy')
    state_dec_1 = np.load(args.path + '/extra_states/state_dec_1.npy')
    state_dec_2 = np.load(args.path + '/extra_states/state_dec_2.npy')
    state_fea_1 = np.load(args.path + '/extra_states/state_fea_1.npy')
    state_fea_2 = np.load(args.path + '/extra_states/state_fea_2.npy')

    if args.motion == 'flow' or args.motion == 'flow_mc':
        flow = np.load(args.path + '/extra_states/pre_flow.npy')

state_encoder = tf.nn.rnn_cell.LSTMStateTuple(state_enc_1, state_enc_2)
state_decoder = tf.nn.rnn_cell.LSTMStateTuple(state_dec_1, state_dec_2)
state_feature = tf.nn.rnn_cell.LSTMStateTuple(state_fea_1, state_fea_2)

# if args.motion == 'flow' or args.motion == 'flow_mc':
frame_input = tf.concat([frame_1, frame_2, flow], axis=-1)
frame_output, state_encoder, state_decoder, state_feature, flow \
    = nn.get_network_pp(frame_input, state_encoder, state_decoder, state_feature, args.motion, args.input_norm)

s_enc_1, s_enc_2 = state_encoder
s_dec_1, s_dec_2 = state_decoder
s_fea_1, s_fea_2 = state_feature

saver = tf.train.Saver(max_to_keep=None)
save_root = './model/Extrapolation'
save_path = save_root + '/lambda_' + str(args.l) + '_extra/'
# latest_model = tf.train.latest_checkpoint(checkpoint_dir=save_path)
print("\033[31m" + save_path + "\033[0m")
if os.path.exists(save_path + 'model.ckpt.index'):
    saver.restore(sess, save_path + 'model.ckpt')
else:
    saver.restore(sess, save_path + 'model.ckpt-150000')

frame_out, state_enc_1, state_enc_2, \
state_dec_1, state_dec_2, state_fea_1, state_fea_2, pre_flow \
    = sess.run([frame_output, s_enc_1, s_enc_2, s_dec_1, s_dec_2, s_fea_1, s_fea_2, flow])

# np.save(args.path + 'f' + str(args.idx).zfill(3) + '_extra.npy', frame_out)
frame_out = np.uint8(np.round(np.clip(frame_out, 0, 1) * 255.0))
imageio.imwrite(args.path + 'f' + str(args.idx).zfill(3) + '_extra.png', frame_out[0])

np.save(args.path + '/extra_states/state_enc_1.npy', state_enc_1)
np.save(args.path + '/extra_states/state_enc_2.npy', state_enc_2)
np.save(args.path + '/extra_states/state_dec_1.npy', state_dec_1)
np.save(args.path + '/extra_states/state_dec_2.npy', state_dec_2)
np.save(args.path + '/extra_states/state_fea_1.npy', state_fea_1)
np.save(args.path + '/extra_states/state_fea_2.npy', state_fea_2)
np.save(args.path + '/extra_states/pre_flow.npy', pre_flow)

