Our other works on learned video compression:

- Perceptual Learned Video Compression (PLVC) (**IJCAI 2022**) [[Paper](https://arxiv.org/abs/2109.03082)]

- Hierarchical Learned Video Compression (HLVC) (**CVPR 2020**) [[Paper](https://arxiv.org/abs/2003.01966)] [[Codes](https://github.com/RenYang-home/HLVC)]

- Recurrent Learned Video Compression (RLVC) (**IEEE J-STSP 2021**) [[Paper](https://ieeexplore.ieee.org/abstract/document/9288876)] [[Codes](https://github.com/RenYang-home/RLVC)]

- OpenDVC: An open source implementation of DVC [[Codes](https://github.com/RenYang-home/OpenDVC)] [[Technical report](https://arxiv.org/abs/2006.15862)]

# Advancing Learned Video Compression with In-loop Frame Prediction

The project page for the paper:

> Ren Yang, Radu Timofte and Luc Van Gool, "Advancing Learned Video Compression with In-loop Frame Prediction", IEEE Transactions on Circuits and Systems for Video Technology (IEEE T-CSVT), 2022. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9950550)

If our paper and codes are useful for your research, please cite:
```
@article{yang2022advancing,
  title={Advancing Learned Video Compression with In-loop Frame Prediction},
  author={Yang, Ren and Timofte, Radu and Van Gool, Luc},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  publisher={IEEE}
}
```

If you have questions or find bugs, please contact:

Ren Yang @ ETH Zurich, Switzerland   

Email: r.yangchn@gmail.com

## Codes

### Preperation

We feed RGB images into the our encoder. To compress a YUV video, please first convert to PNG images with the following command.

```
ffmpeg -pix_fmt yuv420p -s WidthxHeight -i Name.yuv -vframes Frame path_to_PNG/f%03d.png
```

Note that, our RLVC codes currently only support the frames with the height and width as the multiples of 16. Therefore, when using these codes, if the height and width of frames are not the multiples of 16, please first crop frames, e.g.,

```
ffmpeg -pix_fmt yuv420p -s 1920x1080 -i Name.yuv -vframes Frame -filter:v "crop=1920:1072:0:0" path_to_PNG/f%03d.png
```

We uploaded a prepared sequence *BasketballPass* here as a test demo, which contains the PNG files of the first 100 frames. 

### Dependency

- Tensorflow 1.12
  
  (*Since we train the models on tensorflow-compression 1.0, which is only compatibable with tf 1.12, the pre-trained models are not compatible with higher versions.*)

- Tensorflow-compression 1.0 ([Download link](https://github.com/tensorflow/compression/releases/tag/v1.0))

  (*After downloading, put the folder "tensorflow_compression" to the same directory as the codes.*)
  
- SciPy 1.2.0

  (*Since we use misc.imread, do not use higher versions in which misc.imread is removed.*)

- Pre-trained models ([Download link](https://data.vision.ee.ethz.ch/reyang/ALVC/model/model.zip))

  (*Download the folder "model" to the same directory as the codes.*)

- VTM ([Download link](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM))

  (*In our PSNR model, we use VVC to compress I-frames. Please compile VTM and put the folder "VVCSoftware_VTM" in the same directory as the codes.*)
  
### Test code

The augments in the ALVC test code (ALVC.py) include:

```
--path, the path to PNG files;

--l, lambda value. The pre-trained PSNR models are trained by 4 lambda values, i.e., 256, 512, 1024 and 2048, with increasing bit-rate/PSNR. 
```
For example:
```
python ALVC.py --path BasketballPass
```
