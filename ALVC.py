import os
import argparse

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--path", default='BasketballPass')
parser.add_argument("--l", type=int, default=512, choices=[256, 512, 1024, 2048])
args = parser.parse_args()

frames = len(os.listdir(args.path))

print('Running ALVC with extrapolation (P-frames)')
os.system('python Recurrent_AutoEncoder_Extrapolation.py --path ' + args.path + ' --frame ' + str(frames)
          + ' --l ' + str(args.l))

print('Running entropy coding')
os.system('python Recurrent_Prob_Model.py --path ' + args.path + ' --frame ' + str(frames)
          + ' --l ' + str(args.l))

print('Running ALVC with interpolation (B-frames)')
os.system('python Interpolation_Compression.py --path ' + args.path + ' --frame ' + str(frames)
          + ' --l ' + str(args.l))