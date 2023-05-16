# DRT: A Lightweight Single Image Deraining Recursive Transformer
By Yuanchu Liang, Saeed Anwar, Yang Liu </br>
College of Engineering and Computer Science, The Australian National University
</br>
Data61, CSIRO
</br>

---

This repo is the official implementation of DRT: A Lightweight Single Image Deraining Recursive Transformer ([arxiv](https://arxiv.org/abs/2204.11385)).  
The paper has been accepeted by [NTIRE2022](https://data.vision.ee.ethz.ch/cvl/ntire22/) workshop of [CVPR2022](https://cvpr2022.thecvf.com/).
</br>

### Abstract
> Over parameterization is a common technique in deep learning to help models learn and generalize sufficiently to the given task; nonetheless, this often leads to enormous network structures and consumes considerable computing resources during training. Recent powerful transformer-based deep learning models on vision tasks usually have heavy parameters and bear training difficulty. However, many dense-prediction low-level computer vision tasks, such as rain streak removing, often need to be executed on devices with limited computing power and memory in practice. Hence, we introduce a recursive local window-based self-attention structure with residual connections and propose deraining a recursive transformer (DRT), which enjoys the superiority of the transformer but requires a small amount of computing resources. In particular, through recursive architecture, our proposed model uses only ~1.3% of the number of parameters of the current best performing model in deraining while exceeding the state-of-the-art methods on the Rain100L benchmark by at least 0.33 dB. Ablation studies also investigate the impact of recursions on derain outcomes. Moreover, since the model contains no deliberate design for deraining, it can also be applied to other image restoration tasks. Our experiment shows that it can achieve competitive results on desnowing. The source code and pretrained model can be found at https://github.com/YC-Liang/DRT.

![DRT Network Architecture](https://github.com/YC-Liang/DRT/blob/main/Images/Network.png)

### Running
**Installations**</br>
*Note the script is only tested on Ubuntu 20.04*.
* Download `Python3`, script was ran by `Python 3.7.11`
* Install `cuda >= 10.1` with `cudnn >= 7`.
* Install `PyTorch >= 1.7.1` with `torchvision >= 0.8.2`
* Install `timm >= 0.3.2`
* Install `torch-summary >= 0.8.2`

**Data Sets**</br>
*Training*: download the following data sets and unzip into the `Data` folder </br>
* `Rain800`: [click here](https://github.com/hezhangsprinter/ID-CGAN).
* `Rain100L-Train`: [click here](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)
* `Rain100H-Train`: [click here](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html)
* `Snow100K-training`: [click here](https://sites.google.com/view/yunfuliu/desnownet)

*Testing*: inside the `Data` folder, run `mkdir Test-HiNet`. 
* Unzip `Rain800`, `Rain100L` and `Rain100H` into the `Test-HiNet` folder: [click here](https://github.com/megvii-model/HINet)
* Unzip `Snow100K-testset` into the `Data` folder: [click here](https://sites.google.com/view/yunfuliu/desnownet)

The `Data` folder should looks something like this:</br>
```
|Data
|--Rain800
    |-- training
        |-- ...
    |-- ...
|--Rain100L-Train
    |-- rain
        |-- ...
    |-- norain
        |-- ...
|--Rain100H-Train
    |-- rain
        |-- ...
    |-- norain
        |-- ...
|--Snow100K-training
    |-- all
        |...
|--Test-HiNet
    |-- Rain100H
        |-- ...
    |-- Rain100L
        |-- ...
    |-- Test100
        |-- ...
|--Snow100K-testset
    |-- Snow100K-L
        |-- ...
    |-- Snow100K-M
        |-- ...
    |-- Snow100K-S
        |-- ...
```

**Running Instructions**</br>
The core content is written in the Jupyter Notebook `DRT.ipynb`.</br>
To evaluate, change the evaluate folder directory in the `Load Data` section and set the last line of the notebook to </br>
* `run(train_net = False, loadCkp = True, loadBest = True, new_dataset = False)` </br>

To train from scratch, change the training directory in the `Load Data` section and set the last line of the notebook to
* `run(train_net = True, loadCkp = False, loadBest = False, new_dataset = False)`

### Results
**Quantitative Results**
![PSNR and SSIM Results on Three Data Sets](https://github.com/YC-Liang/DRT/blob/main/Images/PSNR_and_SSIM.png)

**Rain100L**
![Rain100L](https://github.com/YC-Liang/DRT/blob/main/Images/Rain100L.png)

**Rain100H**
![Rain100H](https://github.com/YC-Liang/DRT/blob/main/Images/Rain100H.png)

**Test100**
![Test100](https://github.com/YC-Liang/DRT/blob/main/Images/Test100.png)

**Realistic**  
![Realistic](https://github.com/YC-Liang/DRT/blob/main/Images/Real.png)


### Citations
```
@InProceedings{Liang_2022_CVPR,
    author    = {Liang, Yuanchu and Anwar, Saeed and Liu, Yang},
    title     = {DRT: A Lightweight Single Image Deraining Recursive Transformer},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {589-598}
}
```
```
@misc{https://doi.org/10.48550/arxiv.2204.11385,
  doi = {10.48550/ARXIV.2204.11385},
  url = {https://arxiv.org/abs/2204.11385},
  author = {Liang, Yuanchu and Anwar, Saeed and Liu, Yang},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {DRT: A Lightweight Single Image Deraining Recursive Transformer},
  publisher = {arXiv},
  year = {2022}, 
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```



