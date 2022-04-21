# SGDL

<div align=center>
<img src=./fig/overview.jpg width="70%" ></img>
</div>


This repository is the Pytorch implementation of the paper:

> Yunjun Gao, Yuntao Du, Yujia Hu, Lu Chen, Xinjun Zhu, Ziquan Fang and Baihua Zheng. (2022). Self-Guided Learning to Denoise for Robust Recommendation. Paper in [ACM DL]() or Paper in [arXiv](https://arxiv.org/abs/2204.06832). In SIGIR'22, Madrid, Spain, July 11-15, 2022.

## Introduction

Self-Guided Denoising Learning (SGDL) is a new denoising paradigm which is able to collect memorized interactions at the early stage of the training,  and leverage those data as denoising signals to guide the following training of the model in a meta-learning manner. Besides, SGDL can automatically switch its learning phase at the memorization point from memorization to self-guided learning, and select clean and informative memorized data via an adaptive denoising scheduler to further improve the robustness.

## Citation

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{SGDL22,
  title		= {Self-Guided Learning to Denoise for Robust Recommendation},
  author	= {Yunjun Gao and 
  	           Yuntao Du and 
  	           Yujia Hu and 
  	           Lu Chen and 
  		   Xinjun Zhu and 
  		   Ziquan Fang and 
  		   Baihua Zheng},
  booktitle	= {{SIGIR}},
  year		= {2022}
}
```

## Environment Requirements

+ Ubuntu OS
+ Python >= 3.7.9
+ torch  1.4.0+
+ Nvidia GPU with cuda 10.1+

## Datasets

Three popular public datasets for recommendation are used in our research:

+ MovieLens-100K
+ [Adressa](https://www.adressa.no/)
+ [Yelp](https://www.yelp.com/dataset/challenge)

## Reproducibility & Training

To demonstrate the reproducibility of the best performance reported in our paper and facilitate researchers to track whether the model status is consistent with ours, we provide the best parameter settings (might be different for the customized datasets) in the scripts, and provide [the log for our trainings](./code/log).

+ MovieLens-100k with LightGCN

```shell
python main.py --batch_size 128 --lr 0.0005 --meta_lr 0.0005 --model lgn --eval_freq 5 --stop_step 8 --dataset ml100k
```

+ Adressa with LightGCN

```shell
python main.py --batch_size 1024 --lr 0.0005 --meta_lr 0.0005 --model lgn --eval_freq 10 --stop_step 4 --dataset adressa
```

+ Yelp with LightGCN

```shell
python main.py --batch_size 2048 --lr 0.0005 --meta_lr 0.0005 --model lgn --eval_freq 10 --stop_step 4 --dataset yelp
```

+ For other parameters and baselines, please refer to the settings in Section 4.1.4 of our paper.
