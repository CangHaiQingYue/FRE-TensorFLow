# FRE-TensorFLow
This is an implementation of FRE(Edge Detection with Feature Re-extraction Deep Convolutional Neural Network)network, which focus on the Edge Detection.<br>
In this paper, we propose an edge detector based on feature re-extraction (FRE) of a deep convolutional neural network to effectively utilize features extracted from each stage, and design a new loss function. The proposed detector is mainly composed of three modules: backbone, side-output, and feature fusion. The backbone module provides preliminary feature extraction; the side-output module makes network architecture more robustly map features from different stages of the backbone network to edge-pixel space by applying residual learning, and the feature fusion module generates the edge map. Generalization ability on the same distribution is verified using the BSDS500 dataset, achieving optimal dataset scale (ODS) F-score = 0.804. Cross-distribution generalization ability is verified on the NYUDv2 dataset, achieving ODS F- score = 0.701. In addition, we find that freezing backbone network can significantly speed up training process, without much overall accuracy loss (ODS F-score of 0.791 after 5.4k iterations).
## Data Preparation
The BSDS500 dataset and NYUD dataset are available:
```
wget http://mftp.mmcheng.net/liuyun/rcf/data/bsds_pascal_train_pair.lst
wget http://mftp.mmcheng.net/liuyun/rcf/data/HED-BSDS.tar.gz
wget http://mftp.mmcheng.net/liuyun/rcf/data/PASCAL.tar.gz
wget http://mftp.mmcheng.net/liuyun/rcf/data/NYUD.tar.gz
```

To make network easy to training, we should transpose images to TFRecord files.
To Run
```
cd data_procider
python image_to_tfreord
```
## Training Model
To run this model, we should change some path in ```data_loader.py   train.py  test.py``` respectively.
After that, just run 
                          ``` 
                          sh training_fre.sh
                         ```
## Inference
To get the edge maps, one should changes some path and runs:

                          ``` 
                          sh testing_fre.sh
                         ```
## Citations
If you used dataset mentioned above， please cite the following papers:
```
@inproceedings{
  title={Edge Detection with Feature Re-extraction Deep Convolutional Neural Network},
  author={Changbao Wen, Pengli Liu, Wenbo Ma, Zhirong Jian, Changheng Lv, Jitong Hong, Xiaowen Shi},
  journal={Journal of Visual Communication and Image Representation},
  year={2018}
}
```
```
@inproceedings{liu2017richer,
  title={Richer Convolutional Features for Edge Detection},
  author={Liu, Yun and Cheng, Ming-Ming and Hu, Xiaowei and Wang, Kai and Bai, Xiang},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2017}
}
```
and 
```
@inproceedings{xie2015holistically,
  title={Holistically-nested edge detection},
  author={Xie, Saining and Tu, Zhuowen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1395--1403},
  year={2015}
}
```
