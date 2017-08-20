# MNIST_center_loss.pytorch

A pytorch implementation of center loss on MNIST and it's a toy example of ECCV2016 paper [A Discriminative Feature Learning Approach for Deep Face Recognition](https://github.com/ydwen/caffe-face)

In order to ease the classifiers, center loss was designed to make samples in each class flock together.

Results are shown below:

<div align=center><img width="400" height="300" src="https://github.com/jxgu1016/MNIST_center_loss.pytorch/raw/master/images/1.0.jpg"/></div>
<div align=center><img width="400" height="300" src="https://github.com/jxgu1016/MNIST_center_loss.pytorch/raw/master/images/0.jpg"/></div>

`left`:softmax loss and center loss `right`: only softmax loss

The code also incluedes visualization of the training process and please wait until these gifs load

<div align=center><img width="400" height="300" src="https://github.com/jxgu1016/MNIST_center_loss.pytorch/raw/master/images/1.0.gif"/></div>
<div align=center><img width="400" height="300" src="https://github.com/jxgu1016/MNIST_center_loss.pytorch/raw/master/images/0.gif"/></div>
