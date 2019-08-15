# CNN_glareClassification
Some experiments with CNNs and OpenCV for glare classification


Used transfer learning on famous CNN models such as VGG and Imagenet to classiy glare in different types of images.

The accuracy obtained(around 75%) is far better than the best glare identifier(Evalglare 60-65%) from HDR images.

 HDRs are 32 bit images, which usually dont work with Pillow thats bundled with  
 Keras flow_from_directory() hence OpenCV is used.
