# A Deep Multi-Level Network for Saliency Prediction
This repository contains reference code for computing Multi-Level Net (ML-Net) saliency maps based on the following paper:

*Marcella Cornia, Lorenzo Baraldi, Giuseppe Serra, Rita Cucchiara. "A Deep Multi-Level Network for Saliency Prediction." In Proceedings of the 23rd International Conference on Pattern Recognition, 2016.*

Please cite with the following Bibtex code:

```
@inproceedings{mlnet2016,
  author = {Cornia, Marcella and Baraldi, Lorenzo and Serra, Giuseppe and Cucchiara, Rita},
  title = {{A Deep Multi-Level Network for Saliency Prediction}},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  year = {2016}
}
```

An arXiv pre-print is available at this [link](https://arxiv.org/abs/1609.01064).

## Abstract

This paper presents a novel deep architecture for saliency prediction. Current state of the art models for saliency prediction employ Fully Convolutional networks that  perform a non-linear combination of features extracted from the last convolutional layer to predict saliency maps. We propose an architecture which, instead, combines features extracted at different levels of a Convolutional Neural Network (CNN). Our model is composed of three main blocks: a feature extraction CNN, a feature encoding network, that weights low and high level feature maps, and a prior learning network. 
We compare our solution with state of the art saliency models on two public benchmarks datasets. Results show that our model outperforms under all evaluation metrics on the SALICON dataset, which is currently the largest public dataset for saliency prediction, and achieves competitive results on the MIT300 benchmark.

![mlnet-fig](https://raw.githubusercontent.com/marcellacornia/mlnet/master/figs/mlnet.jpg)

## Usage

To compute saliency maps using our pre-trained model:
```
python main.py test path/to/images/folder/
```
where "path/to/images/folder/" is the path of a folder containing the images for which you want to calculate the saliency maps.

To train our model from scratch:
```
python main.py train
```
Beside, it is necessary to set parameters and paths in the [config.py](config.py) file.

* Weights VGG-16: [vgg16_weights.h5](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing) 
* Weights ML-Net SALICON: [mlnet_salicon_weights.pkl](https://drive.google.com/file/d/0B3ZguV08iwjsOGFEWlRfZkVqaWs/view?usp=sharing) 

## Requirements
* [Theano](https://github.com/Theano/Theano) 0.9.0
* [Keras](https://github.com/fchollet/keras) 1.1.0, configured for using Theano as backend 
* OpenCV 3.0.0

Note: Be sure to have ```"image_dim_ordering": "th"``` and ```"backend": "theano"``` in your keras.json file.

## Precomputed Saliency Maps

We provide saliency maps predicted by our network for two standard datasets:
* [SALICON validation set](https://drive.google.com/file/d/0B3ZguV08iwjsYS1jTC16TEFQbU0/view?usp=sharing) 
* [MIT1003 dataset](https://drive.google.com/file/d/0B3ZguV08iwjsM2RjTk5HYjExTUE/view?usp=sharing) 
  
## Contact

For more datails about our research please visit our [page](http://imagelab.ing.unimore.it/imagelab/researchActivity.asp?idActivity=30).

If you have any general doubt about our work, please use the [public issues section](https://github.com/marcellacornia/mlnet/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:marcella.cornia@unimore.it> or <mailto:lorenzo.baraldi@unimore.it>.
