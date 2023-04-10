

运行前先activate conda 环境

`jupyter notebook --ip 外网ip地址` 然后就可以查看服务器上的ipynb 文件了. 

vscode 双击就可以折叠cell. 还可以用outline

## Assignment1

\# TODO: how does the initialization changes the result

\# TODO: how should we choose the learning rate and the iterations

\# TODO: what happens when we initialize at 0 

When start point =-4.00, the loss will decrease slowly. Even in 100 iteration, the loss will not converage.  And the figure shows it is not reach the minimum value.

When start point =-1.00, the loss will decrease very quickly. In about 10 iteration, the loss will converage.  

When start point = 0.00,  we cannot find anything. Because in this point, the derivative(the x.grad) is nan.

When start point = 1.00,  the loss will decrease very quickly. Because in this point, the derivative(the x.grad) is nan.

When start point = 4.00,  the loss will decrease slowly. Because in this point, the derivative(the x.grad) is nan.

When learning rate = 0.1

| start point | loss                                | converage iteration | find min? | reason                                   |
| ----------- | ----------------------------------- | ------------------- | --------- | ---------------------------------------- |
| -4          | decrease slowly                     | >100                | x         | Iteration num is not enough              |
| -1          | decrease very quickly               | ~10                 | √         | close to minimum                         |
| 0           | nan                                 | nan                 | x         | the derivative at 0 (the x.grad) is nan. |
| 1           | the loss will decrease very quickly | ~10                 | x         | local minima                             |
| 4           | the loss will decrease slowly       | >100                | x         | local minima                             |

When we increase learning rate = 0.2

| start point | loss                                | converage iteration | find min? | reason                                   |
| ----------- | ----------------------------------- | ------------------- | --------- | ---------------------------------------- |
| -4          | decrease slowly                     | 80                  | √         | Iteration num is enough                  |
| -1          | decrease very quickly               | ~5                  | √         | close to minimum                         |
| 0           | nan                                 | nan                 | x         | the derivative at 0 (the x.grad) is nan. |
| 1           | the loss will decrease very quickly | ~5                  | x         | local minima                             |
| 4           | the loss will decrease slowly       | >100                | x         | local minima                             |

The learning rate controls how much the model's parameters are updated during each iteration. If the learning rate is too high, the model may overshoot the optimal parameters and fail to converge. If the learning rate is too low, the model may converge too slowly or get stuck in a local minimum. 

We can use a small learning rate, such as 0.01 or 0.001, and adjust it as needed during training. 

Iterations: The number of iterations, or epochs, determines how many times the algorithm will loop through the entire dataset during training. If the number of iterations is too low, the model may not converge to the optimal parameters. If the number of iterations is too high, the model may overfit to the training data and perform poorly on new, unseen data. A common approach is to monitor the performance of the model on a validation set during training and stop training when the performance stops improving or starts to worsen.



#### 正则化

\# learn more about regularization: https://stats.stackexchange.com/a/18765

\# tricks like (dropout, weight decay, learning rate scheduling) might help

\# see what others have tried https://stats.stackexchange.com/q/376312

To add regularization to a PyTorch neural network model, you can use various techniques, including L1 or L2 regularization, dropout, early stopping, and others.

#### weight decay



#### pooling

你就可以理解为卷积核每空两格做一次卷积，卷积核的大小是2x2， 但是卷积核的作用是取这个核里面最大的值（即特征最明显的值），而不是做卷积运算

减小计算量, 学习特征. 



多加一层    nn.Conv2d(256, 256, kernel_size=3, padding=1),

​    nn.ReLU(inplace=True),好像效果不大. 从99.26到99.28

​        \# read about http://wikipedia.org/wiki/Data_augmentation

​        \# take a look here https://pytorch.org/vision/stable/transforms.html

​        \# add some randomaization to improve the generalization

​        \# at least use 3 other transforms



#### 数据增强

notebook修改了之后要找到第一次定义的cell 重新往下运行, 不能从中间运行. 不然会很奇怪. 



#### 验收

问问题

为什么0没有?  因为函数undefined. 不能除以0 

为什么用torch.ones_like(ys)? 因为要归一化? 听不懂

为什么要nn.Module? 因为要维护parameter list. 它这个都是唯一答案非常恶心. 

用的是什么loss? criterion=nn.MSELoss()

用的是什么dataset?  hog 数据集.

要仔细把notebook读一遍背诵一下. 

不需要运行

## Assignment2

num_classes = 3

不能x.cuda(), 需要赋值

对于 Penn-Fudan 数据集，这是一个带有标记对象的图像数据集，可以使用的一些可能的指标是：

1. Mean Intersection over Union (mIoU)：这是评估对象检测和分割模型准确性的常用指标。它测量预测和地面实况分割掩码之间的重叠，并定义为预测和地面实况掩码的交集与并集的比率。
2. F1 分数：这是评估对象检测和分割模型的另一个常用指标。它衡量模型的准确率和召回率之间的平衡，定义为准确率和召回率的调和平均值。
3. 准确性：该指标衡量正确分类对象的比例，通常用于分类任务。

选择指标时，重要的是要考虑它应该是训练指标还是验证指标。训练指标衡量模型在训练数据上的表现，而验证指标衡量模型在保留验证集上的表现。通常，建议使用验证指标，因为这可以更准确地衡量模型的泛化性能。但是，在训练期间跟踪训练指标以监视模型的进度并检测潜在问题（例如过度拟合）也很有用。

### confusion matrix

行是actual class, 列是 predicted class.

是猫, 预测是狗, false negative.  type I error

是狗, 预测是猫, false positive.   type 2 error.

是狗, 预测不是狗, true negative. 

precision , tp/tp +fp .  预测猫对的/  **预测是猫的数量**

accuracy, tp + tn / (total)  (预测猫对+ 预测狗对)/  total

recall: TP/ (TP+FN)   预测猫对的/  **是猫的数量**

precision相当于查准，可以理解为“我 预测是猫中有多少是对的”

召回率相当于查全，可以理解为猫中，我下载到了多少”

F.cross_entropy(x, y, w) 需要传入weight,  因为 standard cross-entropy loss function may be biased towards the majority class, 

in binary classification, the accuracy for the Positive class is the same as the accuracy for the Negative class.

f1同时考虑了recall和precision. 

penn fudan dataset,  annotation是一个mask,  像素一一对应,  mask 是unsigned int mask (所以后面要改成long), 原图是3 channel RGB图. 

ax.imshow(data_item['mask'], alpha=0.5) alpha是透明度

miinst clutter dataset.

-1给背景, [0,9] 给实际数字. 

#### part1

penn fudan, 只考虑行人和background.   image和mask是pil 格式. 

mask是instance segmentation, 我们要改成semantic segment.

`__getattr__` 包装了dataset所以可以直接访问属性

把padding的变成特殊的. 

#### task2

用FCN,  U shape,  因为有 encoder和decoder. 

空间downsampling, 用pooling . 或者卷积.

upsampling, 用 un pooling , or deconv. deconv一般效果比较好. 

网络太深, 会导致 梯度vanish. 所以, 需要skip或者leakyrelu. 不然收敛很慢. 

Transpose2d会增加分辨率, deconv.   **interpolation**

 kernel size of 2x2. 1一个像素变 2x2 output feature map. 

训练了fudan

[1,2]weight,  0.66 loss, 变成0.56.  [1,1,] loss好像更小了. recall更小了,  metric都变差了. 

f1同时考虑了recall和precision. 

##### FCN-8S

用pretrain的 作为encoder. 比如VGG16 imagenet训练过,  frozenweight, fine tune only shallow decoder part, transfer learning.

In a 1x1 convolution, the kernel size is 1x1 and the stride is also 1x1.

The output of the convolution operation is simply the element-wise product of the weights and the input.

### detection

 MNISTClutter数据集

考试: iou 为什么不用? 因为iou不可导. 

右x, y 下 , h,w.

anchor :An anchor is just a bounding box ($4$ values) with $C$ class label probabilities (logits) and an additional confidence score $p_c$, sometimes called objectness score. 

we can have for each grid cell three anchors; one for tall objects, one for wide objects, and one for square objects (we can define more)

yolov3

non maximum suppression. 解决double box. 这个果然考了. 

每个数据处理都要变换的操作是什么 一个是crop, 一个是normalizetion. 

Collc fn 可以处理各种数据. 

## project3

翻译是many to many.

训练单个recurrent cell.每个element都会经过这个cell.

cell 会输入一个hidden state.

但是, 有时候语序需要变换. 所以要用bidirectional RNN. 

activation 用 tanh

LSTM , 有memory ,忘掉无关的.The gate does this by applying a sigmoid function to the weighted sum of the previous hidden state and the current input, and then element-wise multiplying the result with the memory cell. 再加一个cell state ,  可以应对长序列, 

GRU   计算更快. batch_first=True 这样可以输入batch放前面他内部会转换. `nn.GRU` uses a number of `nn.GRUCell`'s according to the `num_layers` and the `bidirectional` arguments.

### part2

cifar10, 训练的到53上不去了. batch size变大,  hidden size变大. 

把图分成16x16patches. So total 256 tokens.

加个残差, 准确率高了18%, 太强了. 

### part3

transformer可以并行处理整个序列. 而且不区分顺序. 最重要的是多头自注意力机制. 

why only use x[:, 0, :]

 The output of the Transformer encoder is a sequence of hidden states for each patch in the input image. However, the final prediction for the image classification task only requires a single vector. Therefore, the CLS token (**the first token of the sequence)** is extracted by selecting the **first element of the output tensor** along the second dimension (which **represents the sequence length**). This extracted token is then passed through a linear layer (`self.head`) to produce the final classification prediction.

简而言之，注意力只是一种权衡每个序列元素对其他序列元素影响的方法。例如，在视觉变换器（ViT）中，输入图像被裁剪成16x16像素的斑块，作为一个序列传递给网络（每个斑块都是一个序列元素，斑块在原始图像中的位置作为位置编码）。这样一来，每个补丁都可以从第一层开始关注其他每个补丁。感受区是整个图像，注意力权重矩阵直观地代表了每个斑块对其他斑块进行图像分类的重要性（回顾一下，注意力矩阵的大小是$N\times N$）。

我们实质上是从序列中提取此 . 序列中的第一个patch,通过一个layer来预测. 

位置编码用于基于 Transformer 的模型，以添加有关每个标记在输入序列中的位置的信息。由于这些模型没有任何固有的词序或位置概念，因此这些附加信息对于模型准确处理输入序列是必需的。

输入序列中每个标记的位置使用每个位置的固定向量进行编码。该向量在传递给 Transformer 模型之前被添加到相应的标记嵌入中。生成的向量包含有关令牌及其在序列中的位置的信息。

一个简单的可学习位置编码，而不是一个固定的位置编码，可以用来为位置编码引入一些灵活性。该模型不是使用固定编码，而是在训练期间为每个位置学习最佳编码。这可以帮助模型更好地适应手头任务的具体情况。

然而，值得注意的是，可学习的位置编码会带来额外的计算开销，并且可能需要更多的训练数据才能获得良好的性能。在某些情况下，固定位置编码可能就足够了，而且效率更高。

#### 问题

1. attr.shape 是 4,100,100.  4 是batch size, 100是feature size
2. kernel size变成3, 那要在四周 padding 1 .
3. r i 的shape是什么?chunk分了之后  [batch size, hiddensize] ,  15是hidden size *3 
4. GRU 每一层输出的shape是什么?   out = [batch , sequence , hidden size ] , hidden = [sequence x bidirected, batch size , hidden size ]
5. 还有什么是线性不变的?  linear. 什么是线性不变? 
6. confusion matrix是什么样的.
7. softmax后 和是 1 . bmm 相乘

