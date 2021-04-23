# Pose2Seg running on your dataset
I do some job to make Pose2Seg run on my dataset.

You can git clone this repo and follow the instructions following to make Pose2Seg run on your dataset or single image.

This programme is still very naive. I just use this programe for test and get a poor result in my application scenarios. So I will not give more effort on this programme, unless someone have some interest on it.

## 1 git clone AlphaPose
Note that, Pose2Seg need images and pose as its input. So, if you don't have any keypoints result on your images, or your keypoints result may not satisfy Pose2Seg, you should first git clone [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose). You can use this programme to get a COCO Format 17 keypoints of you images. 

You can get some instructions in its Readme file.

## 2 build a conda environment
You can use the same environment as Pose2Seg. And you can get my env in pose2seg.yml.


## 3 transform AlphaPose result format to Pose2Seg format
After get the keypoints result from AlphaPose, you can use xxx.py to transform the AlphaPose result to Pose2Seg format result.


## 4 place your images and pose file
Before get segmentation result of Pose2Seg, you should place your images and pose file in the folder named ```Pose2Seg/data/```, the folder instructure is as below.
```
data
├── coco2017 
│ ├── annotations
│ │ ├── person_keypoints_val2017_pose2seg.json 
│ ├── val2017
│ │ ├── ####.jpg
```
As you can see, I do some change from Pose2Seg folder instructure. Because, I don't need any training file for test its proformance in my dataset.

## 5 do testing
You can get into ```Pose2Seg/``` and use the command below to run this test on your dataset:
```
python test.py --weights pose2seg_release.pkl --coco
```
You will see output result in  ```Pose2Seg/output``` folder. The segment part will be showed with **red color** as following:

\
![example](figures/785.png)

You can change this color in test.py in line 49.

>
># Pose2Seg
>
>*Official* code for the paper "Pose2Seg: Detection Free Human Instance Segmentation"[[ProjectPage]](http://www.liruilong.cn/projects/pose2seg/index.html)[[arXiv]](https://arxiv.org/abs/1803.10683) @ CVPR2019.
>
>The *OCHuman dataset* proposed in our paper is released [here](https://github.com/liruilong940607/OCHumanApi)
>
><div align="center">
><img src="figures/pipeline.jpg" width="1000px"/>
><p> Pipeline of our pose-based instance segmentation framework.</p>
></div>
>
>## Setup environment
>
>``` bash
>pip install cython matplotlib tqdm opencv-python scipy pyyaml numpy
>pip install torchvision torch
>
>cd ~/github-public/cocoapi/PythonAPI/
>python setup.py build_ext install
>cd -
>```
>
>## Download data
>
>- COCO 2017
>    - [COCO 2017 Train images [118K/18GB]](http://images.cocodataset.org/zips/train2017.zip)
>    - [COCO 2017 Val images [5K/1GB]](http://images.cocodataset.org/zips/val2017.zip)
>    - [COCOPersons Train Annotation (person_keypoints_train2017_pose2seg.json) [166MB]](https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_train2017_pose2seg.json)
>    - [COCOPersons Val Annotation (person_keypoints_val2017_pose2seg.json) [7MB]](https://github.com/liruilong940607/Pose2Seg/releases/download/data/person_keypoints_val2017_pose2seg.json)
>    
>- OCHuman
>    - [images [667MB] & annotations](https://cg.cs.tsinghua.edu.cn/dataset/form.html?dataset=ochuman)
>    
>**Note**: 
>`person_keypoints_(train/val)2017_pose2seg.json` is a subset of `person_keypoints_(train/val)2017.json` (in [COCO2017 Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)). We choose those instances with both keypoint and segmentation annotations for our experiments.
>
>## Setup data
>
>The `data` folder should be like this:
>
>    data  
>    ├── coco2017
>    │   ├── annotations  
>    │   │   ├── person_keypoints_train2017_pose2seg.json 
>    │   │   ├── person_keypoints_val2017_pose2seg.json 
>    │   ├── train2017  
>    │   │   ├── ####.jpg  
>    │   ├── val2017  
>    │   │   ├── ####.jpg  
>    ├── OCHuman 
>    │   ├── annotations  
>    │   │   ├── ochuman_coco_format_test_range_0.00_1.00.json   
>    │   │   ├── ochuman_coco_format_val_range_0.00_1.00.json   
>    │   ├── images  
>    │   │   ├── ####.jpg 
>
>## How to train
>
>``` bash
>python train.py
>```
>
>**Note**: Currently we only support for single-gpu training.
>
>## How to test
>
>This allows you to test the model on (1) COCOPersons val set and (2) OCHuman val & test set.
>
>``` bash
>python test.py --weights last.pkl --coco --OCHuman
>```
>
>We retrained our model using this repo, and got similar results with our paper. The final weights can be download [here](https://drive.google.com/file/d/193i8b40MJFxawcJoNLq1sG0vhAeLoVJG/view?usp=sharing).
>
>## About Human Pose Templates in COCO
>
><div align="center">
><img src="figures/pose_templates.png" width="500px"/>
><p> Pose templates clustered using K-means on COCO.</p>
></div>
>
>This repo already contains a template file `modeling/templates.json` which was used in our paper. But you are free to explore different cluster parameters as discussed in our paper. See [visualize_cluster.ipynb](visualize_cluster.ipynb) for an example.
>
>
