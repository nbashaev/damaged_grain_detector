# Damaged grain detector

The task is to estimate the amount of broken grain in a given photo.
Training set contains pairs of pictures like this:

![](/examples/train_led.png "led")

![](/examples/train_uv.png "uv")

The second shot is made using ultraviolet light. Every broken region is initially marked with special paint.

This problem was formalized as a segmentation problem: for each pixel one needs to determine whether it belongs to a broken region or not. To solve this problem we used classical segmentation neural network [U-Net](https://arxiv.org/abs/1505.04597) and different loss functions for highly unbalanced segmentation. Training process is organized using Python and deep learning framework MXNet.

Trained model is optimized using NVIDIA TensorRT. Inference is fully implemeted in C++. One can see an example of model prediction below.

![](/examples/pic_1.png "Initial picture")

![](/examples/pred_1.png "Model prediction")
