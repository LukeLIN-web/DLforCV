# DLforCV

Briefly discuss the contributions of each of the two papers above. Afterwards, compare ResNet
and AlexNet in terms of performance and architectural design

## AlexNet

### Contribution

1. This paper has made a significant contribution by focusing on supervised learning, unlike previous deep learning efforts that mostly pursued unsupervised learning as supervised learning was believed to be less effective and similar to SVM. However, the results of this paper show that with larger networks and longer training, the performance of supervised learning can be greatly improved.
2. The end-to-end dataset was processed without any pre-processing, just trimmed to 256x256 and trained on raw RGB.
3. They used multiple GPUs for training and split the model across them.
4. They used SGD, which at the time was less popular compared to more stable algorithms like GD, but it was later found that the noise in SGD had positive effects on generalization. SGD then became widely adopted, as did the momentum they used for optimization.

### performance: 

1. In the ILSVRC-2012 competition, AlexNet achieved a top-5 test error rate of 15.3%, which was a winning performance compared to the second-best entry's 26.2%.

2. On the test data, they achieved top-1 and top-5 error rates of 37.5% and 17.0%, better than SOTA
3. The vectors produced by Alexnet are well represented in the semantic space and can be used to cluster similar images together.

### architectural design

1. The neural network consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax. It has a total of 60 million parameters and 650,000 neurons.

2. To improve training efficiency, we utilized non-saturating neurons and an optimized GPU implementation of the convolution operation. Non-saturating neurons help to prevent the vanishing gradient or gradient explosion problem.
3. To reduce overfitting in the fully-connected layers they employed a recently-developed regularization method called “dropout” that proved to be very effective.  Dropout is a technique where outputs from some neurons are randomly set to zero during training, resulting in a new model each time.  They put dropout on the first two fully connected layers.  Without the dropout, the model would be severely overfitted, but with the dropout it would be twice as slow as others.  Later,  it was thought that dropout was a regular term, equivalent to L2. CNNs do not use such a large fully-connected layer, and dropout is not commonly used anymore.  Dropout is more often used in RNN and attention.
4. overlapping pooling.
5. With ReLu, training is faster, and no normalization is required to avoid saturation.
6. The first convolutional layer has two part, one on each of the two GPUs, two GPU does not communicate. The output of the second convolutional layer is available on both GPU0 and GPU1, and the fourth and fifth convolutional layers two GPUs do not communicate.
7. The first fully connected layer, the input is the output of the fifth convolutional layer of each GPU combined. At the end, each GPU has a vector of length 2048, the whole vector length is 4096, this vector can represent semantic information, if the vector is similar, then the two images may be images of the same object.
8. Each layer is randomly initialized with a Gaussian distribution with mean = 0 and standard deviation of 0.01. For large models such as bert's standard deviation, 0.02 can be used.

#### 数据集处理

不做任何预处理, 只是剪裁到256*256.  在raw RGB上训练. 

#### 防止过拟合

1. 数据增强, 把一些图片人工变大. 用CPU, 当时数据增强计算量不大, 
2. PCA  主成分分析, 通道上变换. 让图片和原始图片不太一样.

#### 学习率

alexnet是 当validatoin error不变的时候, 就把learning rate  乘0.1. 之后很多也是这么做的. 不过太陡峭了, 现在用cos 来下降学习率. 学习率从0开始, 慢慢上升, 慢慢下降. 

resnet是 120个epoch, 每30轮下降0.1  

## ResNet

### 贡献

1. proposes residual learning framework to traing DNN. reformulat the layers as learning residual functions with reference. The accuracy of the network does not deteriorate even if the number of layers increases.  The result is a network that is easy to train and delivers high accuracy.
2. The paper demonstrates the application of the residual learning framework by training a 1202-layer DNN on the CIFAR-10 dataset.

### performance

1. design a 152-layer deep network, which is 8 times deeper than the VGG network, while maintaining a lower computational complexity.
2. The network achieved a top 5 error rate of 3.57% on the ImageNet test set, securing first place in the 2015 image classification competition.
3. By replacing the backbone network with ResNet, this paper were able to achieve a 28% improvement in object detection accuracy on the COCO dataset.

### architectural design

1.  Instead of hoping each few stacked layers fit a underlying mapping, we let these layers fit a residual mapping F(x) = H(x) -x. H(x) is  the desired underlying mapping.   It means, we add the output of shallow layers to the input of deeper layer. The fomulation can be realized by feedforward neural networks with "shortcut connections". Identity shortcut add neither extra parameter nor computational complexity. 
2. The first layer 7x7 convolution, the second module has 3x3 max pool, 6 convolution layers, one shortcut for each two convolution layers, and finally 1x1 average pool, 1000-d fc , softmax .

| Architectural design |                   |          |
| -------------------- | ----------------- | -------- |
|                      | Alexnet           | ResNet   |
| dropout              | Yes               | No       |
| momentum             | 0.9               | 0.9      |
| weight decay         | 0.0005            | 0.0001   |
| full connect layer   | 3                 | no       |
| Important Idea       | ReLu,multiple GPU | shortcut |

### 比较

The first performance, achieved by AlexNet, is a notable one in the field of deep learning and computer vision. AlexNet achieved a top-5 test error rate of 15.3% in the ILSVRC-2012 competition, outperforming the second-best entry's 26.2% error rate. The network also showed strong results on test data, with top-1 and top-5 error rates of 37.5% and 17.0%, respectively, which were better than the state-of-the-art (SOTA) at the time. Additionally, the representations produced by AlexNet were well-formed in the semantic space and could be used to cluster similar images together.

The second performance, achieved by the network described in the paper, is also significant in the field of deep learning and computer vision. The network is 8 times deeper than the VGG network, with 152 layers, and has a lower computational complexity. The network achieved a top-5 error rate of 3.57% on the ImageNet test set, securing first place in the 2015 image classification competition. By replacing the backbone network with ResNet, the authors were able to achieve a 28% improvement in object detection accuracy on the COCO dataset.

In comparison, the second performance is generally better than the first one in terms of accuracy, as the top-5 error rate is lower and the improvement in object detection accuracy on the COCO dataset is higher. However, both performance results were important milestones in the development of deep learning and demonstrate the potential of these techniques for image classification and object detection tasks.

