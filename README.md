# Siamese-NAS: Using Trained Samples Efficiently to Find Lightweight Neural Architecture by Prior Knowledge

![](https://img.shields.io/badge/Python-3-blue)
![](https://img.shields.io/badge/TensorFlow-2-orange)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

This project provides a predictor-based NAS method, and it is namely Siamese-Predictor. The siamese-predictor is constructed with the proposed Estimation Code, which is the prior knowledge about the training procedure. In our experiments, the proposed siamese-predictor surpasses the current SOTA predictor (BRP-NAS) on NASBench-201. We also propose the search space Tiny-NanoBench for lightweight CNN architecture. This well-designed search space is easier to find better architecture with few FLOPs than NASBench-201. 

Paper Link: Coming soon !

<div align=center>
<img src=https://github.com/D0352276/Siamese-NAS/blob/main/demo/siams_predictor.png width=100% />
</div>

<div align=center>
<img src=https://github.com/D0352276/Siamese-NAS/blob/main/demo/nsam.png width=100% />
</div>

## Requirements
- [Python 3](https://www.python.org/)
- [TensorFlow 2](https://www.tensorflow.org/)
- [OpenCV](https://docs.opencv.org/4.5.2/d6/d00/tutorial_py_root.html)
- [Numpy](http://www.numpy.org/)
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [NASBench-201](https://github.com/D-X-Y/NAS-Bench-201)



## Training and Evaluation for Paper Results
There are various predictors and search spaces in our paper, and if you want to train the specific predictor, you just change line 140 of file  "train_eval.py". For example, you can train the Siamese-Predictor +NSAM +BTS on NASBench-201 (CIFAR-100).
```bash
if(__name__=="__main__"):
    model_type="siams +NSAM +BTS"
    data_type="nasbench201_cifar100" 
    ......
```
If you want to train the naive predictor on Tiny-NanoBench, you just rewrite line 140 to  this.
```bash
if(__name__=="__main__"):
    model_type="predictor"
    data_type="tiny_nanobench" 
    ......
```
In default setting, every traing and evaluation procedure will repeat 100 runs, if you want to change it, please change the arg "params_dict["repeats"]" at line 134.


## Complete Search Spaces
Attached to this project is already most of the searc space, it contains NASBench-201 (CIFAR-10), NASBench-201 (CIFAR-100) and Tiny-NanoBench. More search spaces are coming soon!  Due to the large number of files, you have to unzip it yourself.


## More Info for Tiny-NanoBench 
We record the relationship between FLOPs and accuracy for NASBench-201. We can find that it consists of several groups at the FLOPs axis. Based on this observation, we set a constraint of about 35 MFLOPs to focus on the smallest FLOPs group (blue group in part (a) of Fig. 4), we call this subset Tiny-NASBench-201. We believe that Tiny-NASBench-201 is not the optimal search space at the tiny FLOPs ( 35M) level, so we proposed the Tiny-NanoBench trained on CIFAR-10.
<div align=center>
<img src=https://github.com/D0352276/Siamese-NAS/blob/main/demo/tinynanobench.png width=100% />
</div>


## Results
Fig. 6 shows the experiments on NASBench-201, which is fixed N or K and evaluates these predictors. Part (a) of Fig. 6 is fixed N to 30, then gradually increases K to draw the entire curve. Fig. 7 shows that the siamese-predictor searching on Tiny-NanoBench can find the architecture of higher accuracy than Tiny-NASBench-201 when N is large.
<div align=center>
<img src=https://github.com/D0352276/Siamese-NAS/blob/main/demo/results.png width=90% />
</div>

## TODOs
-  More intuitive API.
- ...

