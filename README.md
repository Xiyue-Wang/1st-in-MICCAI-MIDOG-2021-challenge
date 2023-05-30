
## A Generalizable and Robust Deep Learning Algorithm for Mitosis Detection in Multicenter Breast Histopathological Images(*Medical Image Analysis*)


[Journal Link](https://www.sciencedirect.com/science/article/abs/pii/S1361841522003310)

This is the source code for the first place solution to the [MICCAI 2021 MIDOG microscopy domain generalization challenge](https://midog2021.grand-challenge.org/evaluation/midog-final-test-phase/leaderboard/). Our [Algorithms and Weights](https://grand-challenge.org/algorithms/mitos/) is already on the platform, ready to run directly, you can get the results by applying, on your image.Please note that your grand challenge account has to be certified by the school or business, you will have a green tick once you pass it, so we can agree, then you can upload the image and run the code.

Please open new threads or address all questions to xiyue.wang.scu@gmail.com
#### Hardware

* 32GB of RAM
* 4*Nvidia V100 32G GPUs

## Updates / TODOs
Please follow this GitHub for more updates.
- [ ] Add training code
- [X] Add inference code for evaluation.
- [X] Add model.
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
python get_fda_image.py
```

#### 3.Get instance mask
please see the [HoVer-Net](https://github.com/vqdang/hover_net),get the cell mask.And then intersect with the mitotic bbox to get the mitotic mask, and finally preprocess 512*512 patches.
[Here](https://drive.google.com/drive/folders/1WrB3Mu_rLtSKbWfbwv2JmOYHSqZDvUof?usp=sharing) is our processed some image and the corresponding mask



#### Inference
test image is on the path ./test/007.tiff

```
python process.py
```


## License

This code(FMDet) is released under the GPLv3 License and is available for non-commercial academic purposes.

### Citation
Please use below to cite this [paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841522003310) if you find our work useful in your research.


```
@article{WANG2022102703,
title = {A generalizable and robust deep learning algorithm for mitosis detection in multicenter breast histopathological images},
author = {Xiyue Wang and Jun Zhang and Sen Yang and Jingxi Xiang and Feng Luo and Minghui Wang and Jing Zhang and Wei Yang and Junzhou Huang and Xiao Han},
journal = {Medical Image Analysis},
pages = {102703},
year = {2022},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2022.102703}
}
``` 







