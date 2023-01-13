import numpy as np
import tensorflow as tf
import os
from scipy import misc
from ms_ssim_np import MultiScaleSSIM
import arithmeticcoding

def configure(args, path_root, path_raw):

    if args.l == 256:
        I_level = 37
    elif args.l == 512:
        I_level = 32
    elif args.l == 1024:
        I_level = 27
    elif args.l == 2048:
        I_level = 22

    elif args.l == 8:
        I_level = 3
    elif args.l == 16:
        I_level = 4
    elif args.l == 32:
        I_level = 5
    elif args.l == 64:
        I_level = 6

    path = args.path + '/'

    # if args.mode == 'MS-SSIM':
    #     path_com = path_root + args.path + '_SSIM_' + str(args.l) + '/frames/'
    #     path_bin = path_root + args.path + '_SSIM_' + str(args.l) + '/bitstreams/'
    #     path_lat = path_root + args.path + '_SSIM_' + str(args.l) + '/latents/'
    # else:
    path_com = path_root + args.path + '_' + args.mode + '_' + str(args.l) + '/frames/'
    path_bin = path_root + args.path + '_' + args.mode + '_' + str(args.l) + '/RD_results/'
    path_lat = path_root + args.path + '_' + args.mode + '_' + str(args.l) + '/latents/'

    os.makedirs(path_com, exist_ok=True)
    os.makedirs(path_bin, exist_ok=True)
    os.makedirs(path_lat, exist_ok=True)

    F1 = misc.imread(path_raw + 'f001.png')
    Height = np.size(F1, 0)
    Width = np.size(F1, 1)
    batch_size = 1
    Channel = 3

    if (Height % 16 != 0) or (Width % 16 != 0):
        raise ValueError('Height and Width must be a mutiple of 16.')

    activation = tf.nn.relu

    GOP_size = args.f_P + args.b_P + 1 + args.inter
    GOP_num = int(np.floor((args.frame - 1)/GOP_size))

    return I_level, Height, Width, batch_size, \
           Channel, activation, GOP_size, GOP_num, \
           path, path_com, path_bin, path_lat

def configure_gan(args):

    I_level = args.quality

    path = args.path
    path_video = '/srv/beegfs-benderdata/scratch/reyang_data/data/RLVC/GAN_results_inter/' + args.path + '_' + str(I_level) + '_0.001_rc1'

    # path_com = './GAN_nrec/' + args.path + '_' + str(I_level) \
    #            + '_' + str(args.w_g) + '_rc' + str(args.rc) + '/frames/'
    # path_bin = './GAN_nrec/' + args.path + '_' + str(I_level) \
    #            + '_' + str(args.w_g) + '_rc' + str(args.rc) + '/bitstreams/'
    # path_lat = './GAN_nrec/' + args.path + '_' + str(I_level) \
    #            + '_' + str(args.w_g) + '_rc' + str(args.rc) + '/latents/'

    path_com = path_video + '/frames/'
    path_bin = path_video + '/bitstreams/'
    path_lat = path_video + '/latents/'

    os.makedirs(path_com, exist_ok=True)
    os.makedirs(path_bin, exist_ok=True)
    os.makedirs(path_lat, exist_ok=True)

    F1 = misc.imread('/srv/beegfs-benderdata/scratch/reyang_data/data/RLVC/' + path + '/f001.png')
    Height = np.size(F1, 0)
    Width = np.size(F1, 1)
    batch_size = 1
    Channel = 3

    if (Height % 16 != 0) or (Width % 16 != 0):
        raise ValueError('Height and Width must be a mutiple of 16.')

    activation = tf.nn.relu

    GOP_size = args.f_P + args.b_P + args.inter + 1
    GOP_num = int(np.floor((args.frame - 1)/GOP_size))

    return I_level, Height, Width, batch_size, \
           Channel, activation, GOP_size, GOP_num, \
           path, path_com, path_bin, path_lat

def configure_decoder(args):

    path = '/srv/beegfs02/scratch/reyang_data/data/origCfP/' + args.path + '/'
    path_com = './results/' + args.path + '_' + args.mode + '_' + str(args.l) + '/frames_dec/'
    path_bin = './results/' + args.path + '_' + args.mode + '_' + str(args.l) + '/bitstreams/'
    path_lat = './results/' + args.path + '_' + args.mode + '_' + str(args.l) + '/latents_dec/'

    os.makedirs(path_com, exist_ok=True)
    os.makedirs(path_lat, exist_ok=True)

    activation = tf.nn.relu

    GOP_size = args.f_P + args.b_P + 1
    GOP_num = int(np.floor((args.frame - 1)/GOP_size))

    return activation, GOP_size, GOP_num, \
           path, path_com, path_bin, path_lat


def encode_I(args, frame_index, I_level, path, path_com, path_bin):

    if args.mode == 'PSNR':

        if args.VTM == 1:

            F1 = misc.imread(path + '/f001.png')
            Height = np.size(F1, 0)
            Width = np.size(F1, 1)

            # path_yuv = './RLVC_VTM/' + args.path + '_' + args.mode + '_' + str(args.l) + '/frames/'
            #
            # if not os.path.exists(path_com + 'f' + str(frame_index).zfill(3) + '.yuv'):
            #
            os.system('ffmpeg -i ' + path + 'f' + str(frame_index).zfill(3) + '.png '
                      '-pix_fmt yuv444p ' + path + 'f' + str(frame_index).zfill(3) + '.yuv -y -loglevel error')
            os.system(
                '/scratch_net/maja_second/VVCSoftware_VTM/bin/EncoderAppStatic -c /scratch_net/maja_second/VVCSoftware_VTM/encoder_intra_vtm.cfg '
                '-i ' + path + 'f' + str(frame_index).zfill(3) + '.yuv -b ' + path_bin + 'f' + str(frame_index).zfill(3) + '.bin '
                '-o ' + path_com + 'f' + str(frame_index).zfill(3) +  '.yuv -f 1 -fr 2 -wdt ' + str(Width) + ' -hgt ' + str(Height) +
                ' -q ' + str(I_level) + ' --InputBitDepth=8 --OutputBitDepth=8 --OutputBitDepthC=8 --InputChromaFormat=444 > /dev/null')

            # os.system('cp ' + './RLVC_VTM_extra_7/' + args.path + '_' + args.mode + '_' + str(args.l) + '/bitstreams/' + 'f' + str(frame_index).zfill(3) + '.bin ' +
            #             path_bin + 'f' + str(frame_index).zfill(3) + '.bin')
            # os.system('cp ' + './RLVC_VTM_extra_7/' + args.path + '_' + args.mode + '_' + str(args.l) + '/frames/' + 'f' + str(frame_index).zfill(3) + '.png ' +
            #             path_com + 'f' + str(frame_index).zfill(3) + '.png')
            os.system(
                'ffmpeg -f rawvideo -pix_fmt yuv444p -s ' + str(Width) + 'x' + str(Height) +
                ' -i ' + path_com + 'f' + str(frame_index).zfill(3) + '.yuv '
                + path_com + 'f' + str(frame_index).zfill(3) + '.png -y -loglevel error')

        else:
            os.system('bpgenc -f 444 -m 9 ' + path + 'f' + str(frame_index).zfill(3) + '.png '
                      '-o ' + path_bin + 'f' + str(frame_index).zfill(3) + '.bin -q ' + str(I_level))
            os.system('bpgdec ' + path_bin + 'f' + str(frame_index).zfill(3) + '.bin '
                  '-o ' + path_com + 'f' + str(frame_index).zfill(3) + '.png')

    elif args.mode == 'MS-SSIM':
        os.system(args.python_path + ' ' + args.CA_model_path + '/encode.py --model_type 1 '
                  '--input_path ' + path + 'f' + str(frame_index).zfill(3) + '.png' +
                  ' --compressed_file_path ' + path_bin + 'f' + str(frame_index).zfill(3) + '.bin'
                  + ' --quality_level ' + str(I_level))
        os.system(args.python_path + ' ' + args.CA_model_path + '/decode.py --compressed_file_path '
                  + path_bin + 'f' + str(frame_index).zfill(3) + '.bin'
                  + ' --recon_path ' + path_com + 'f' + str(frame_index).zfill(3) + '.png')

    # bits = os.path.getsize(path_bin + str(frame_index).zfill(3) + '.bin')
    # bits = bits * 8

    F0_com = misc.imread(path_com + 'f' + str(frame_index).zfill(3) + '.png')
    F0_raw = misc.imread(path + 'f' + str(frame_index).zfill(3) + '.png')

    F0_com = np.expand_dims(F0_com, axis=0)
    F0_raw = np.expand_dims(F0_raw, axis=0)

    if args.metric == 'PSNR':
        mse = np.mean(np.power(np.subtract(F0_com / 255.0, F0_raw / 255.0), 2.0))
        quality = 10 * np.log10(1.0 / mse)
    elif args.metric == 'MS-SSIM':
        quality = MultiScaleSSIM(F0_com, F0_raw, max_val=255)

    print('Frame', frame_index, args.metric + ' =', quality)

    return quality

def hific_I(args, frame_index, I_level, path, path_com, path_bin):

    # python = '/scratch_net/maja_second/miniconda3/envs/env_cpu_3/bin/python3.6'
    # os.system(python + ' tfci.py compress hific-' + I_level + ' ' + path + ' ' + path_bin)
    # os.system(python + ' tfci.py decompress ' + path_bin + ' ' + path_com)
    #
    path_hific = '/scratch_net/maja_second/compression/models/hific_results/'
    #
    # if I_level == 'hhi':
    #     I_level = 'hi'

    os.system('cp ' + path_hific + path + '_' + str(I_level) + '/f' + str(frame_index).zfill(3) + '.tfci ' + path_bin)
    os.system('cp ' + path_hific + path + '_' + str(I_level) + '/f' + str(frame_index).zfill(3) + '.png ' + path_com)

    F0_com = misc.imread(path_com + 'f' + str(frame_index).zfill(3) + '.png')
    F0_raw = misc.imread(path + '/f' + str(frame_index).zfill(3) + '.png')

    F0_com = np.expand_dims(F0_com, axis=0)
    F0_raw = np.expand_dims(F0_raw, axis=0)

    if args.metric == 'PSNR':
        mse = np.mean(np.power(np.subtract(F0_com / 255.0, F0_raw / 255.0), 2.0))
        quality = 10 * np.log10(1.0 / mse)
    elif args.metric == 'MS-SSIM':
        quality = MultiScaleSSIM(F0_com, F0_raw, max_val=255)

    bits = os.path.getsize(path_bin + '/f' + str(frame_index).zfill(3) + '.tfci')
    bits = bits * 8 / np.size(F0_com, 1) / np.size(F0_com, 2)

    print('Frame', frame_index, 'bpp =', bits, args.metric + ' =', quality)

    return quality, bits

def decode_I(args, frame_index, path_com, path_bin):

    if args.mode == 'PSNR':
        os.system('bpgdec ' + path_bin + 'f' + str(frame_index).zfill(3) + '.bin '
                  '-o ' + path_com + 'f' + str(frame_index).zfill(3) + '.png')

    elif args.mode == 'MS-SSIM':
        os.system(args.python_path + ' ' + args.CA_model_path + '/decode.py --compressed_file_path '
                  + path_bin + 'f' + str(frame_index).zfill(3) + '.bin'
                  + ' --recon_path ' + path_com + 'f' + str(frame_index).zfill(3) + '.png')

    print('Decoded I-frame', frame_index)


def entropy_coding(frame_index, lat, path_bin, latent, sigma, mu):

    if lat == 'mv':
        bias = 50
    else:
        bias = 100

    bin_name = 'f' + str(frame_index).zfill(3) + '_' + lat + '.bin'
    bitout = arithmeticcoding.BitOutputStream(open(path_bin + bin_name, "wb"))
    enc = arithmeticcoding.ArithmeticEncoder(32, bitout)

    for h in range(latent.shape[1]):
        for w in range(latent.shape[2]):
            for ch in range(latent.shape[3]):
                mu_val = mu[0, h, w, ch] + bias
                sigma_val = sigma[0, h, w, ch]
                symbol = latent[0, h, w, ch] + bias

                freq = arithmeticcoding.logFrequencyTable_exp(mu_val, sigma_val, np.int(bias * 2 + 1))
                enc.write(freq, symbol)

    enc.finish()
    bitout.close()

    bits_value = os.path.getsize(path_bin + bin_name) * 8

    return bits_value


def entropy_decoding(frame_index, lat, path_bin, path_lat, sigma, mu):

    if lat == 'mv':
        bias = 50
    else:
        bias = 100

    bin_name = 'f' + str(frame_index).zfill(3) + '_' + lat + '.bin'
    bitin = arithmeticcoding.BitInputStream(open(path_bin + bin_name, "rb"))
    dec = arithmeticcoding.ArithmeticDecoder(32, bitin)

    latent = np.zeros([1, mu.shape[1], mu.shape[2], mu.shape[3]])

    for h in range(mu.shape[1]):
        for w in range(mu.shape[2]):
            for ch in range(mu.shape[3]):

                mu_val = mu[0, h, w, ch] + bias
                sigma_val = sigma[0, h, w, ch]

                freq = arithmeticcoding.logFrequencyTable_exp(mu_val, sigma_val, np.int(bias * 2 + 1))
                symbol = dec.read(freq)
                latent[0, h, w, ch] = symbol - bias

    bitin.close()

    np.save(path_lat + '/f' + str(frame_index).zfill(3) + '_' + lat + '.npy', latent)
    print('Decoded latent_' + lat + ' frame', frame_index)

    return latent




