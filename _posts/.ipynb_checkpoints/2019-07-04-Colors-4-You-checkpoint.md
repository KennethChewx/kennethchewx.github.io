---
layout: post
title: "Learning to color pictures with DcGANs - Capstone for Data Science Immersive Course"
---
    
## My introduction in Computer Vision
----
    
Hi everyone, this is going to be my first post ever, and I would just like to say that although I did not start out doing data science-ish posts with your usual basic machine learning stuff like iris dataset, titanic etc, I am still very much a beginner in this field. Still, I'm almost 3/4 done with my data science immersive course at General Assembly in Singapore and would love to contribute what I have learnt thus far about ML and deep learning.
    
I was greatly intrigued with the use of neural networks in computers, as the topic of neural networks greatly aligned with the field of cognitive neuroscience during my undergraduate time. Of course, this led me to my interest in the field computer vision within data science. For starters, I wanted to try and color gray pictures with deep learning! This was an interesting topic to myself because I am a light novel and comic reader myself. Having colored images and colored comics are always a plus point to making reading much enjoyable. Plus, colors give amazing perspective to readers too.


<p align="center">
  <img src="http://www.onepiecepodcast.com/wp-content/uploads/2018/01/color-810x466.png">    
</p>
*One piece manga full colored page*
    
## Data collection and data processing
----
        
To start off, I would first have to get a large dataset of colored pictures that can be used to put my model into training. After searching, I found the dataset from a FloydHub user `emilwallner` who shared his dataset of 9,294 images that was collected from unsplash. These pictures were all very high in quality and were all standardized in size with 256x256. Thinking about the minor requirement that the capstone project had (which was to at least have 1,500 rows of data), I was thrilled about using this dataset. 

<p align="center">
  <img src="https://miro.medium.com/max/100/0*_ppsR4sRdQqFw4Y5">    
</p>
*Image taken from unsplash from emilwallner dataset*
    
To make things easier, emilwallner also did a similar project and had a nice introduction into coloring pictures using CNN with less than 100 lines of codes! For more information, you could click [here][emil]. Already, this medium post gives a great introduction into neural networks and also explanations code by code what each line does. So if you are not already familiar with it, you could drop there to take a look. 
    
So I started off my coding by emulating what was already out there in the internet. Suffice to say, those less than 100 line of neural network was a great starting point for me to build my own neural network. I learnt how to decode and load images into numpy arrays, learnt how that input shapes are important and that batch sizes entering the model had to be optimized to ensure that my computer memory doesn’t run out. Most importantly, I learnt that we could break down my colorization problem into something else, which was to turn it into a conversion from RGB color space to LAB color space. This greatly coincided with my knowledge too that in normal human vision, our optical nerves in our eyes are mostly connected to rods which is used to detect high spatial acuity, while cones which provide the color perception are much lesser in numbers. Furthermore, to closely mimic the human eye, images that hit the retina are actually inverted. So, included within the step of preprocessing, one of the conditions was to randomly flip images upside down, so that the model can also learn better.

<p align="center">
  <img src="https://qph.fs.quoracdn.net/main-qimg-f543dfb3879656e214d40d16a5b6ff17">    
</p>
*Objects viewed are inverted on the retina*
    
In addition, there were other image preprocessings that I did to the input images. This came easy with keras' `ImageDataGenerator`, where I could easily create unlimited number of datapoints/images from the same input based on certain parameters that is passed in the [data generator][data generator]. One of the more interesting augmentation was the [histogram stretching][histogram stretching] method. This method would allow one to bring out the contrast of the image by plotting the density of contrast on a histogram. Thus, pixels which does not conform to a particular density within the area will be adjusted accordingly. This becomes especially useful for my input images since it is largely grayscale. The contrast would help the models learn those features better. To implement this function together with the ImageDataGenerator, I simply used skimage's exposure library to randomly apply one of three augmentations that could be done: Adaptive equalization, contrast stretching and histogram equalization. This could be passed through a random number generator and as a function in the *preprocessing_function* parameter.

<p align="center">
  <img src="https://scikit-image.org/docs/0.13.x/_images/sphx_glr_plot_equalize_001.png">   
</p>
*As per skimage's documentation*
    
### Histogram Equalization 
Histogram Equalization increases contrast in images by detecting the distribution of pixel densities in an image and plotting these pixel densities on a histogram. The distribution of this histogram is then analyzed and if there are ranges of pixel brightnesses that aren’t currently being utilized, the histogram is then “stretched” to cover those ranges, and then is “back projected” onto the image to increase the overall contrast of the image.

### Contrast Stretching 
Contrast Stretching takes the approach of analyzing the distribution of pixel densities in an image and then “rescales the image to include all intensities that fall within the 2nd and 98th percentiles.”

### Adaptive Equalization 
Adaptive Equalization differs from regular histogram equalization in that several different histograms are computed, each corresponding to a different section of the image; however, it tends to over-amplify noise in otherwise uninteresting sections.

Lastly for preprocessing, to reduce the time taken for the model to learn those important features when passing through the convolution layers into that latent variable space, I [normalized][normalized] the LAB values to according to their color space. I divided the channels in LAB by [100, 128 and 128] respectively. This is done so that the gradients can be adjusted quicker during back propagation. By making the range to relatively the same, we ensure that the gradient descent can converge much faster.
    
## Convolution Neural Network model 
----

Choosing to use Convolutional Neural Networks (CNN) was an easy conclusion for me because I felt that CNN was greatly applicable in this case. Although CNN is one of many forms of neural networks, it has been heavily used in Computer Vision. A typical CNN consists of convolutional layers, pooling layers, fully connected layers, and normalization layers. To understand how CNN work, there are [posts][posts] that can help answer those burning questions. Where a classification problem is concern, pooling and fully connected layers will help to create a decent latent variable space to sieve out features that are more defining towards certain classes of images and thus prediction of a higher probability of those classes. However, for my colorization problem, I would not want to lose any of the image quality and thus, will not be using those layers. Instead, I will include batch normalization and dropout layers, which are regularization layers to prevent overfitting of my CNN model when learning to color the output images. 

Also as a start to building a CNN model, I found it to be safer to follow grounds which were previously explored, and I chose to follow the architecture from EmilWallner's neural network. It seems to work well when training the model on few images (less than 200). Looking at his final version, I've also discovered interesting model architectures that were proven to give some amazing results. For example in the work of [Federico, Diego & Lucas (2017)][Federico, Diego & Lucas (2017)], they incorporated a pre-existing CNN model on top of their own model using inception-resnet-V2.
    
<p align="center">
  <img src="https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/7aeaeaeb09c605963ff04b63ade03dbeb5555f45/4-Figure1-1.png">    
</p>
*CNN + Inception-ResNet-V2 architecture*
    
Also, there were ideas before the above architecture, as pursued by Satoshi and his team in 2016, where the idea was to concatenate repeat vectors from different levels of feature extraction after passing through several convolutional layers.

<p align="center">
  <img src="https://raw.githubusercontent.com/2021rahul/colorization_tensorflow/master/img/Architecture.jpg">    
</p>
*Satoshi Iizuka, Edgar Simo-Serra, and Hiroshi Ishikawa architecture (2016)*
    
Judging from empirical evidence in their papers that both neural networks seems to work pretty well, I decided to give it a try and incorporate something similar to their works.

{% highlight ruby %}
model = Sequential()
    
#Downsampling batch
model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3),activation='relu', padding='same'))                     
model.add(BatchNormalization())                                               #(bs,256, 256,64)
model.add(Conv2D(64, (3, 3), activation='relu',padding='same', strides=2))            
model.add(BatchNormalization())                                               #(bs,128,128,64)
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))                     
model.add(BatchNormalization())                                                #(bs,128,128,128)
model.add(Conv2D(128, (3, 3), activation='relu',padding='same', strides=2))          
model.add(BatchNormalization())                                                   #(bs,64,64,128)
model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))                     
model.add(BatchNormalization())                                                 #(bs,64,64,256)
model.add(Conv2D(256, (3, 3), activation='relu',padding='same', strides=2))            
model.add(BatchNormalization())                                                    #(bs,32,32,256)
model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))                     
model.add(BatchNormalization())                                                   #(bs,32,32,512)

#Upsampling batch
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))        
model.add(BatchNormalization())
model.add(Dropout(0.5))                                                       #(bs,32,32,512)
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))     
model.add(BatchNormalization())
model.add(Dropout(0.5))                                                       #(bs,32,32,256)
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))        
model.add(BatchNormalization())
model.add(Dropout(0.5))                                                       #(bs,32,32,128)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))        
model.add(BatchNormalization()) 
model.add(UpSampling2D((2, 2)))                                               #(bs,64,64,64)
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))        
model.add(BatchNormalization())
model.add(UpSampling2D((2, 2)))                                               #(bs,128,128,32)
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))         
model.add(UpSampling2D((2, 2)))                                               #(bs,256,256,2)

#Finish model
model.compile(optimizer='rmsprop', loss=ssim_loss ,metrics=['mse','mean_absolute_error'])
{% endhighlight %}
    
Included in this model, I have used Relu activation function for the hidden layers and tanh for the last layer. The use of the tanh is because the output layer which are the ab channels have normalized range between -1 and 1, which coincide with the output of tanh. As for the loss function, I used multi-scale SSIM because as feedbacked from my previous experiments, brown is the most closest color to every oher color in the spectrum. Thus, the output using MSE as loss function turns out to be largely brown. On the other hand, [SSIM][SSIM] helps measure the the picture's similarity as a whole between channels from the target and the generated output. 
    
<p align="center">
  <img src="https://scikit-image.org/docs/dev/_images/sphx_glr_plot_ssim_001.png">    
</p> 

## Results 
    
Unfortunately, after training the above model on 1000 epochs, the output did not turn out well. In retrospect, I overdid on the batch normalization. However, this gave me a reason to pursue other forms of models for my coloring problem and thus I encountered DcGANs.
    
## Deep Convolutional Generative Adversarial Network (DcGAN) 
----

GANs have been the latest trend in deep learning, where instead of 1 model trying to produce the best fit prediction, we instead have 2 models working in adversarial terms trying to fine tune each other. Akin to the analogy of police and thieves; The generator, which is the thief, will try to generate images as close to the ground truth as possible. On the other hand the discriminator, with the role of the police, will try to differentiate between the ground truth and the generated ouput from the generator. This adversarial relationship forces both models to get better over time, and  will help to improve color accuracy of the picture that is generated from the generator.    
    
With the incorporation of both convolutional layers and GANs, we get [DcGANs][DcGANs], which utilises convolutional stride in downsampling the input and transposed convolution for upsampling. This makes DcGANs much simpler model that reduces the complexity of the generator and discriminator without causing bottleneck in image quality.
    
<p align="center">
  <img src="https://gluon.mxnet.io/_images/dcgan.png">    
</p>    
    
Two versions of a similar DcGANs (aka Pix2Pix) were available for use in in my case. The original created by Phillip Isola, but was written with PyTorch, and the latter by afflinelayer, written with Tensorflow. I chose to emulate the latter becauseof the tensorflow compatibility. 
    
In addition, because of the *unavoidable problem of long training time* for deep learning models, I had to inevitably cut down my sample size from 9.2k images to 2.2k images. Hopefully, I would still be able to reach my goal of submission of my capstone before graduation from the course. Furthermore, because most of the pictures fall into either of 3 categories of Landscape, People and Objects, I have chose to also only pick 2.2k (2k in training set and 200 in test) images of people only, so that the model would perform better for predicting colors of grayscale people images. Below are the codes that I have fine-tuned for my capstone project.

I trained the model on 1500 epochs on Google Colab GPU, with an average runtime of 200s per epoch, and training it for a total of 4 days.
    
{% highlight ruby %}
    
# Convu filter to downsample image
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

# Convu filter to upsample image
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result
    
def Generator():
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
          ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
          ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None,None,3])
    x = inputs

  # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
    
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

{% endhighlight %}
    
## Results - 2
    
Below are some of the output from the generator.
    
    
    
[emil]: https://medium.com/@emilwallner/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d
[data generator]:   https://keras.io/preprocessing/image/
[jekyll-talk]: https://talk.jekyllrb.com/
[histogram stretching]: http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
[normalized]: https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029
[posts]: https://towardsdatascience.com/understanding-neural-networks-from-neuron-to-rnn-cnn-and-deep-learning-cd88e90e0a90
[Federico, Diego & Lucas (2017)]: https://arxiv.org/abs/1712.03400
[SSIM]: https://pdfs.semanticscholar.org/3401/02ae4239c8b0e810c04be76b758099f2d3cf.pdf
[DcGANs]: https://arxiv.org/pdf/1511.06434.pdf


