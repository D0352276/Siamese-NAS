# SIAMESE-NAS: USING TRAINED SAMPLES EFFICIENTLY TO FIND LIGHTWEIGHT NEURAL ARCHITECTURE BY PRIOR KNOWLEDGE

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

## Training Different Predictors
```bash
#Considering the file "train_eval.py"

if(__name__=="__main__"):
    model_type="siams +NSAM +BTS"
    data_type="nasbench201_cifar100" 
    
    budget_range=[3,20]
    budget_factor=10
    params_dict=GetDefaultParams(model_type,data_type)
    InitResultsDir(params_dict["result_dir"])
    for i in range(budget_range[0],budget_range[1]+1):
        budget=i*budget_factor
        for j in range(params_dict["repeats"]):
            result_dict={}
            max_accs,psp=TrainAndEval(params_dict,budget)
            result_dict["model_type"]=params_dict["model_type"]
            result_dict["budget"]=budget
            result_dict["max_accs"]=max_accs
            result_dict["psp"]=psp
            outpath=params_dict["result_dir"]+"/b"+str(budget)+"_r"+str(j)+".json"
            Dict2JSON(result_dict,outpath)
```
The cfg file can be changed by the paper prosoed versions. It has options like "m2_224_3.cfg", "m2_320_5.cfg", ......, and "vgg_320_9.cfg".


## Fully Model Weights Files
[Download Link](https://drive.google.com/drive/folders/1aF-nK44huxbiQP1zZsg8Mjwfr0n2Nz-k?usp=sharing)

This link contains all weights of CNN models mentioned in the paper.


## Fully Dataset
The entire MS-COCO dataset is too large, here only a few pictures are stored for DEMO, 

if you need complete data, please download on this [page.](https://cocodataset.org/#download)

## Results
 Comparison of SFPN-5 and SFPN-9 with the baseline (SFPN-3) on MS-COCO. Note that the FPS is derived from the I7-6600U.
 
<div align=center>
<img src=https://github.com/D0352276/SFPN-Synthetic-FPN-for-Object-Detection/blob/main/demo/results.png width=60% />
</div>


