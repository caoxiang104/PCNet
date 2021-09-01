# PCNet
Pytorch implementation of "Blind Image Super-Resolution Based on Prior Correction Network"


## Overview
Prior Correction Network
<p align="center"> <img src="figs/model.png" width="100%"> </p>

Kernel Estimate Network
<p align="center"> <img src="figs/KernelEstimate.png" width="80%"></p>

Correction Refinement Network
<p align="center"> <img src="figs/RefineNet.png" width="80%" </p>


## Requirements
- Python 3.6
- PyTorch == 1.7.0
- numpy
- skimage
- imageio
- matplotlib
- cv2


## Train
### 1. Prepare training data 

1.1 Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  dataset and the [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset.

### 2. Begin to train or test
Run `src/demo.sh` to train or test your dataset.



## Citation
```
@article{cao2021blind,
  title={Blind Image Super-Resolution Based on Prior Correction Network},
  author={Cao, Xiang and Luo, Yihao and Xiao, Yi and Zhu, Xianyi and Wang, Tianjiang and Feng, Qi and Tan, Zehan},
  journal={Neurocomputing},
  volume = {463},
  pages = {525-534},
  year={2021},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2021.07.070},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231221011462},
  publisher={Elsevier}
}
```

## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch), [Correction-Filter](https://github.com/shadyabh/Correction-Filter) and [USRNet](https://github.com/cszn/USRNet). We thank the authors for sharing the codes.

