# [WCG-VMamba: A multi-modal classification model for corn disease - ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0168169924012262?via%3Dihub)

## Abstract

> Corn is one of the important food crops and industrial raw materials. However, maize diseases have seriously affected its yield and quality. In order to effectively identify maize diseases, digital image processing technology has been widely used in the agricultural field. The classification of diseases based on digital images enables early detection of corn diseases, reducing farmers’ losses. Although existing corn disease identification methods have made significant progress using deep learning technology for digital image processing, most of these technologies rely on single-modal data for identification and lack the connection between images and texts. To solve this problem, this paper proposes a cross-modal feature alignment fusion model called WCG-VMamba. Firstly, we propose a wavelet visual Mamba (WAVM) network, which integrates the advantages of different visual coding strategies and can reduce the influence of intrinsic noise and other factors on the validity of image features during extraction. Then, we introduce the Cross Modal Alignment Transformer (CMAT), which interacts image features with text features to capture their semantic correlation and determine the weight distribution of image and text features in the fusion process. We then use Transformer coding blocks to fuse features. Finally, the Gaussian Random Walk Duck Swarm Algorithm (GRW-DSA) is proposed to reduce errors in the duck swarm exploration process through Gaussian Random Walk, aiming to find the optimal learning rate. Experiments on self-built datasets and two common datasets show that WCG-VMamba can be effectively used in the task of corn disease classification. Compared with other excellent models such as MobileViT, MobilenetV3, SwinT, and DINOV2, better results have been achieved, our model achieves a recognition accuracy as high as 96.97%, proving its important practical application in promoting agricultural cross-modal models and corn disease control.

## Usage

### 1.Prepare Dataset



### 2.Prepare Enviroment



### 3.Train



#### 4.Evaluate

## acknowledgement

## 🏁Cite Our Work

If our work is useful for your research, please consider citing and give us a star ⭐

```
@article{WANG2025109835,
title = {WCG-VMamba: A multi-modal classification model for corn disease},
journal = {Computers and Electronics in Agriculture},
volume = {230},
pages = {109835},
year = {2025},
issn = {0168-1699},
doi = {https://doi.org/10.1016/j.compag.2024.109835},
url = {https://www.sciencedirect.com/science/article/pii/S0168169924012262},
author = {Haoyang Wang and Mingfang He and Minge Zhu and Genhua Liu},
keywords = {Corn disease, Image-text pairs, Classification, Multimodal, VMamba},
}
```