---
layout: post
title: "Learning to color pictures with DcGANs - Capstone for Data Science Immersive Course"
author: "Kenneth Chew"
meta: "Springfield"
---

<To include a picture here>
    
## My introduction in Computer Vision
----
    
Hi everyone, this is going to be my first post ever, and I would just like to say that although I did not start out doing data science-ish posts with your usual basic machine learning stuff like iris dataset, titanic etc, I am still very much a beginner in this field. Still, I'm almost 3/4 done with my data science immersive course at General Assembly in Singapore and would love to contribute what I have learnt thus far about ML and deep learning.
    
I was greatly intrigued with the use of neural networks in computers, as the topic of neural networks greatly aligned with the field of cognitive neuroscience during my undergraduate time. Of course, this led me to my interest in the field computer vision within data science. For starters, I wanted to try and color gray pictures with deep learning! This was an interesting topic to myself because I am a light novel and comic reader myself. Having colored images and colored comics are always a plus point to making reading much enjoyable. Plus, colors give amazing perspective to readers too.
    
<p align="center">
  <img src="http://www.onepiecepodcast.com/wp-content/uploads/2018/01/color-810x466.png">
    *One piece manga full colored page*    
</p>
        
## Data collection and data processing
----
        
To start off, I would first have to get a large dataset of colored pictures that can be used to put my model into training. After searching, I found the dataset from a FloydHub user *emilwallner* who shared his dataset of 9,294 images that was collected from unsplash. These pictures were all very high in quality and were all standardized in size with 256x256. Thinking about the minor requirement that the capstone project had (which was to at least have 1,500 rows of data), I was thrilled about using this dataset. 

<p align="center">
  <img src="../img/01jzbs.jpg">
    *Image taken from unsplash from emilwallner dataset*    
</p>
    
To make things easier, emilwallner also did a similar project and had a nice introduction into coloring pictures using CNN with less than 100 lines of codes! For more information, you could click [here](https://medium.com/@emilwallner/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d). Already, this medium post gives a great introduction into neural networks and also explanations code by code what each line does. So if you are not already familiar with it, you could drop there to take a look. 
    
So I started off my coding by emulating what was already out there in the internet. Suffice to say, those less than 100 line of neural network was a great starting point for me to build my own neural network. I learnt how to decode and load images into numpy arrays, learnt how that input shapes are important and that batch sizes entering the model had to be optimized to ensure that my computer memory doesn’t run out. Most importantly, I learnt that we could break down my colorization problem into something else, which was to turn it into a conversion from RGB color space to LAB color space. This greatly coincided with my knowledge too that in normal human vision, our optical nerves in our eyes are mostly connected to rods which is used to detect high spatial acuity, while cones which provide the color perception are much lesser in numbers. Furthermore, to closely mimic the human eye, images that hit the retina are actually inverted. So, included within the step of preprocessing, one of the conditions was to randomly flip images upside down, so that the model can also learn better.

<p align="center">
  <img src="../img/main-qimg-f543dfb3879656e214d40d16a5b6ff17.png">
    *Objects viewed are inverted on the retina*    
</p>

In addition, there were other image preprocessings that I did to the input images. This came easy with keras' `ImageDataGenerator`, where I could easily create unlimited number of datapoints/images from the same input based on certain parameters that is passed in the [data generator](https://keras.io/preprocessing/image/). One of the more interesting augmentation was the [histogram stretching](http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm) method. This method would allow one to bring out the contrast of the image by plotting the density of contrast on a histogram. Thus, pixels which does not conform to a particular density within the area will be adjusted accordingly. This becomes especially useful for my input images since it is largely grayscale. The contrast would help the models learn those features better. To implement this function together with the ImageDataGenerator, I simply used skimage's exposure library to randomly apply one of three augmentations that could be done: Adaptive equalization, contrast stretching and histogram equalization. This could be passed through a random number generator and as a function in the *preprocessing_function* parameter.

<p align="center">
  <img src="https://scikit-image.org/docs/0.13.x/_images/sphx_glr_plot_equalize_001.png">
    *As per skimage's documentation*    
</p>
    
###Histogram Equalization
Histogram Equalization increases contrast in images by detecting the distribution of pixel densities in an image and plotting these pixel densities on a histogram. The distribution of this histogram is then analyzed and if there are ranges of pixel brightnesses that aren’t currently being utilized, the histogram is then “stretched” to cover those ranges, and then is “back projected” onto the image to increase the overall contrast of the image.

###Contrast Stretching
Contrast Stretching takes the approach of analyzing the distribution of pixel densities in an image and then “rescales the image to include all intensities that fall within the 2nd and 98th percentiles.”

###Adaptive Equalization
Adaptive Equalization differs from regular histogram equalization in that several different histograms are computed, each corresponding to a different section of the image; however, it tends to over-amplify noise in otherwise uninteresting sections.

Lastly for preprocessing, to reduce the time taken for the model to learn those important features when passing through the convolution layers into that latent variable space, I [normalized](https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029) the LAB values to according to their color space. I divided the channels in LAB by [100, 128 and 128] respectively. This is done so that the gradients can be adjusted quicker during back propagation. By making the range to relatively the same, we ensure that the gradient descent can converge much faster.
    
## Convolution Neural Network model
----

Choosing to use Convolutional Neural Networks (CNN) was an easy conclusion for me because I felt that CNN was greatly applicable in this case. Although CNN is one of many forms of neural networks, it has been heavily used in Computer Vision. A typical CNN consists of convolutional layers, pooling layers, fully connected layers, and normalization layers. To understand how CNN work, there are [posts](https://towardsdatascience.com/understanding-neural-networks-from-neuron-to-rnn-cnn-and-deep-learning-cd88e90e0a90) that can help answer those burning questions. Where a classification problem is concern, pooling and fully connected layers will help to create a decent latent variable space to sieve out features that are more defining towards certain classes of images and thus prediction of a higher probability of those classes. However, for my colorization problem, I would not want to lose any of the image quality and thus, will not be using those layers. Instead, I will include batch normalization and dropout layers, which are regularization layers to prevent overfitting of my CNN model when learning to color the output images. 

Also as a start to building a CNN model, I found it to be safer to follow grounds which were previously explored, and I chose to follow the architecture from EmilWallner's neural network. It seems to work well when training the model on few images (less than 200). Looking at his final version, I've also discovered interesting model architectures that were proven to give some amazing results. For example in the work of [Federico, Diego & Lucas (2017)](https://arxiv.org/abs/1712.03400), they incorporated a pre-existing CNN model on top of their own model using inception-resnet-V2.
    
<p align="center">
  <img src="../img/our_net.png">
    *CNN + Inception-ResNet-V2 architecture*    
</p>
    
Also, there were ideas before the above architecture, as pursued by Satoshi and his team in 2016, where the idea was to concatenate repeat vectors from different levels of feature extraction after passing through several convolutional layers.

<p align="center">
  <img src="https://raw.githubusercontent.com/2021rahul/colorization_tensorflow/master/img/Architecture.jpg">
    *Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa architecture (2016)*    
</p>
    
Judging from empirical evidence in their papers that both neural networks seems to work pretty well, I decided to give it a try and incorporate something similar to their works.
    
    ``` insert code for experiment 4 here ```
    
Included in this model, I have used Relu activation function for the hidden layers and tanh for the last layer. The use of the tanh is because the output layer which are the ab channels have normalized range between -1 and 1, which coincide with the output of tanh. As for the loss function, I used Mean Square Error (MSE), which is a normal loss function in regression problems. Here are some of the outputs from these trials.
