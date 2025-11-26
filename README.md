# VPViT
## 1. Introduction
- This repositary is the official implementation for the paper [Exploiting Class-agnostic Visual Prior for Few-shot Keypoint Detection](). 
- **TL;DR:** A suite of novel methods for few-shot keypoint detection (FSKD), including the visual prior guided vision transformer (VPViT), transductive FSKD, and masking and alignment (MAA), are proposed to enhance keypoint representation learning. The role of VPViT aims to capture the foreground patch relations for keypoint detection. The transductive FSKD is to enhance keypoint representations with unlabeled data and the MAA is investigated to improve detection robustness under occlusions. 
- Tasks handled by our model:
<img src="./pics/VPViT-tasks.png" width="100%"> 

<!-- ## 3. News and upcoming updates

- [x] We initiated the process to open-source codes & models
- [x] ***We will release the source code of VPViT model and VPViT based FSKD upon the acceptance of our paper*** -->

## 2. Requirements
- Python 3.10
- Pytorch 1.12

## 3. VPViT Model
We released the codes of visual prior guided vision transformer (VPViT) which includes the components of Morphology Learner (ML) and Adapted Masked Self-attention (AMSA), as shown in python file ```network/models_fskd_vit.py```.

Moreover, we released the implementation of transductive few-shot keypoint detection, which refines the predicted keypoint by selecting the high-quality pseudo-labels, which can be found in file ```val_transductive.py``` (see functions ```val_transductive()``` and ```iter_transductive()```)

*Since the entire model codes were a bit obsolete after the acceptance of our IJCV paper (two years passed), we only released the most valuable parts related to our paper for your reference. Hope they can inspire your future research. Thanks to your understanding!*


## Citation
If you find our ideas interesting and helpful in your research, please cite our paper. Many thanks!

```
@inproceedings{lu2025exploiting,
  title={Exploiting Class-agnostic Visual Prior for Few-shot Keypoint Detection},
  author={Lu, Changsheng and Zhu, Hao and Koniusz, Piotr},
  booktitle={International Journal of Computer Vision},
  pages={19416--19426},
  year={2025}
}
```

<!-- ## 4. Contact

 * Raise a new [GitHub issue](https://github.com/AlanLuSun/VPViT/issues/new) -->