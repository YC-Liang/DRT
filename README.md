# DRT: A Light Weight Deraining Recursive Transformer
By [Yuanchu Liang](yuanchu.liang@anu.edu.au), [Saeed Anwar](saeed.anwar@anu.edu.au), [Yang Liu](yang.liu3@anu.edu.au)

College of Engineering and Computer Science, The Australian National University

</br>

This repo is the official implementation of DRT: A Lightweight Single Image Deraining Recursive Transformer ([arxiv](https://arxiv.org/abs/2204.11385)).  
The paper has been accepeted by [NTIRE2022](https://data.vision.ee.ethz.ch/cvl/ntire22/) workshop of [CVPR2022](https://cvpr2022.thecvf.com/).

</br>

### Abstract
> Over parameterization is a common technique in deep learning to help models learn and generalize sufficiently to the given task; nonetheless, this often leads to enormous network structures and consumes considerable computing resources during training. Recent powerful transformer-based deep learning models on vision tasks usually have heavy parameters and bear training difficulty. However, many dense-prediction low-level computer vision tasks, such as rain streak removing, often need to be executed on devices with limited computing power and memory in practice. Hence, we introduce a recursive local window-based self-attention structure with residual connections and propose deraining a recursive transformer (DRT), which enjoys the superiority of the transformer but requires a small amount of computing resources. In particular, through recursive architecture, our proposed model uses only ~1.3% of the number of parameters of the current best performing model in deraining while exceeding the state-of-the-art methods on the Rain100L benchmark by at least 0.33 dB. Ablation studies also investigate the impact of recursions on derain outcomes. Moreover, since the model contains no deliberate design for deraining, it can also be applied to other image restoration tasks. Our experiment shows that it can achieve competitive results on desnowing. The source code and pretrained model can be found at https://github.com/YC-Liang/DRT.

### Running
*Codes will be polished and running instructions will be added soon.*

### Results
*Both quantitative and qulitative results will be added here soon.*

### Citations
*CVPR citation to be added*
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



