
## A Generalizable and Robust Deep Learning Algorithm for Mitosis Detection in Multicenter Breast Histopathological Images(This paper is under review)

This is the source code for the first place solution to the [MICCAI 2021 MIDOG microscopy domain generalization challenge](https://midog2021.grand-challenge.org/).

Please open new threads or address all questions to xiyue.wang.scu@gmail.com
#### Hardware

* 32GB of RAM
* 4*Nvidia V100 32G GPUs

## Updates / TODOs
Please follow this GitHub for more updates.
- [ ] Add training code
- [ ] Add inference code for evaluation.
- [X] Add fourier-based data augmentation.
###
#### 1.Preparations
* Data Preparation

   * Download training challenge [MIDOG 2021 data](https://imig.science/midog/download-dataset/)

   * External independent  datasets :  1.[AMIDA13](https://tupac.grand-challenge.org/Dataset/)   2.[MITOSIS14](https://mitos-atypia-14.grand-challenge.org/Dataset/)  3.[TUPAC-auxiliary](https://tupac.grand-challenge.org/Dataset/)  4.[MIDOG2022](https://imig.science/midog/download-dataset/)

  
#### 2.Get fourier-based data [augmentation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf)
 Step 1: Apply FFT to source and target images.

 Step 2: Replace the low frequency part of the source amplitude with that from the target.

 Step 3: Apply inverse FFT to the modified source spectrum.

[Here](https://drive.google.com/drive/folders/1xn0VCAVKFEXzya5bjuoth5vgh0yt9C2Z?usp=sharing) are some images we grabbed randomly

```
get_fda_image.py
```

#### 3.Get instance mask
please see the [HoVer-Net](https://github.com/vqdang/hover_net),get the cell mask.And then intersect with the mitotic bbox to get the mitotic mask, and finally preprocess 512*512 patches.
[Here]() is our processed some image and the corresponding mask


#### Training


#### Inference





## License

This code(FMDet) is released under the GPLv3 License and is available for non-commercial academic purposes.

### Citation
Please use below to cite this paper if you find our work useful in your research.


```
@{wang2022,
  title={A Generalizable and Robust Deep Learning Algorithm for Mitosis Detection in Multicenter Breast Histopathological Images},
  author={Wang, Xiyue and Zhang, Jun and Yang, Sen and Xiang, Jingxi and Feng, Luo and Wang, Minghui and Zhang, Jing  and Yang, Wei and Huang, Junzhou  and Han, Xiao},
  year={2022},
  publisher={Elsevier}
}
``` 







