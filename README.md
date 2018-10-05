### UPDATE(Oct. 2018)
By dropping the bias of the last fc layer according to the [issue](https://github.com/jxgu1016/MNIST_center_loss_pytorch/issues/8), the centers tend to distribute around a circle as reported in the orignal paper.


### UPDATE(May. 2018)
Migration to PyTorch 0.4 done!


### UPDATE(Apr. 2018)
Thanks [@wenfahu](https://github.com/wenfahu) for accomplishing the optimization of *backward()*.


### UPDATE(Mar. 2018)
Problems reported in the [NOTIFICATION](#jump) now has been SOLVED! Functionally, this repo is exactly the same as the official repo. New result is shown below and looks similar to the former one.
If you want to try the former one, please return to [Commits on Feb 12, 2018](https://github.com/jxgu1016/MNIST_center_loss_pytorch/tree/dbeea5380de8a3c6b1b3b3f2c411b980e143dd87).

Some codes can be and should be **optimized** when calculating Eq.4 in [*backword()*](https://github.com/jxgu1016/MNIST_center_loss_pytorch/blob/master/CenterLoss.py) to replace the for-loop and feel free to pull your request.

### NOTIFICATION(Feb. 2018)
<span id="jump"> </span>
In the begining, it was just a practise project to get familiar with PyTorch. Surprisedly, I didn't expect that there would be so many researchers following my repo of center loss. In that case, I'd like to illustrate that this implementation is **not exactly the same** as the official one.

If you read the equations in the paper carefully, the defination of center loss in the Eq. 2 can only lead you to the Eq. 3 but the update equation of centers in Eq. 4 can not be inferred arrcoding to the differentiation formulas. If not specified, the derivatives of one module are decided by the forward operation following the strategy of autograd in PyTorch. Considering the incompatibility  of Eq. 3 and Eq. 4, only one of them can be implemented correctly and what I chose was the latter one. If you remvoe the *centers_count* in my code, this will lead you to the Eq. 3.

This problem exists in other implementaions and the impact remains unknown but looks harmless.

TO DO: To specify the derivatives just like the [original caffe repo](https://github.com/ydwen/caffe-face), instead of being calculated by autograd system.

# MNIST_center_loss_pytorch

A pytorch implementation of center loss on MNIST and it's a toy example of ECCV2016 paper [A Discriminative Feature Learning Approach for Deep Face Recognition](https://github.com/ydwen/caffe-face)

In order to ease the classifiers, center loss was designed to make samples in each class flock together.

Results are shown below:

<div align=center><img width="400" height="300" src="https://github.com/jxgu1016/MNIST_center_loss.pytorch/raw/master/images/1.0-new.jpg"/></div>
<div align=center>softmax loss and center loss(new)</div>
<div align=center><img width="400" height="300" src="https://github.com/jxgu1016/MNIST_center_loss.pytorch/raw/master/images/1.0.jpg"/></div>
<div align=center>softmax loss and center loss(old)</div>
<div align=center><img width="400" height="300" src="https://github.com/jxgu1016/MNIST_center_loss.pytorch/raw/master/images/0.jpg"/></div>
<div align=center>only softmax loss</div>

The code also includes visualization of the training process and please wait until these gifs load

<div align=center><img width="400" height="300" src="https://github.com/jxgu1016/MNIST_center_loss.pytorch/raw/master/images/1.0-new.gif"/></div>
<div align=center>softmax loss and center loss(new)</div>
<div align=center><img width="400" height="300" src="https://github.com/jxgu1016/MNIST_center_loss.pytorch/raw/master/images/1.0.gif"/></div>
<div align=center>softmax loss and center loss(old)</div>
<div align=center><img width="400" height="300" src="https://github.com/jxgu1016/MNIST_center_loss.pytorch/raw/master/images/0.gif"/></div>
<div align=center>only softmax loss</div>
