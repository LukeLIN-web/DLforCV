

运行前先activate conda 环境

在服务器上运行`jupyter notebook --ip 服务器ip地址`  就可以查看服务器上的ipynb 文件了. 

vscode 双击就可以折叠cell. 还可以用outline. vscode新装了软件之后记得reload window, 不然很多识别不出来. 

#### 环境

```
source $(conda info --base)/etc/profile.d/conda.sh
conda create -n cs323 python=3.9.2 -y
conda activate cs323
pip install jupyter_http_over_ws  # for Google Colab
jupyter serverextension enable --py jupyter_http_over_ws  # Google Colab
```

会有问题, np.object弃用了. numpy1.24会出问题. Np.bool ,  tensorboard 源码  conda是垃圾. 能不用就不用.  应该安装比较新的tensorboard. 不要管低版本, 都用高版本就行.

project4要用tensorboard. docker容器要映射网络端口出去. 

pip install tensorboard. conda会无法识别. [ No module named ‘tensorboard’](https://discuss.pytorch.org/t/tensorboard-in-virtualenvironment-no-module-named-tensorboard/77864) 

```
mamba install python=3.9.2 pytorch==1.8.1 torchvision==0.9.1  torchaudio==0.8.1 cudatoolkit=11.1.1 matplotlib=3.3.4 tqdm=4.59.0 tensorboard=2.4.1 numpy=1.23.2 ipykernel==6.19.2 -c pytorch -c conda-forge
```

助教用的是` pytorch1.10.2+python3.9.7+tensorboard2.8+numpy1.21.2`

其实不用tensorboard也行, 就是图很多. 

conda/mamba装pytorch还得指定build, 否则给你装个cpu版的.

要不全pip装最新版的试试.  还是有问题https://github.com/pyvista/pyvista/issues/4380

```
mamba install h5py=2.10.0 -c conda-forge
mamba install pandas -c conda-forge -y
mamba install scikit-learn -c conda-forge -y

mamba install tensorboard

不要用notebook配环境, 把import 提取出来运行python比较快, notebook 每次换了环境都要重启很慢. 
pip3 install torch torchvision torchaudio
pip install notebook
pip install ipykernel
pip install tqdm
pip install h5py
pip install pandas
pip install -U scikit-learn
pip install tensorboard
pip install pyvista==0.35.2

python -m pip freeze  # to see all packages installed in the active virtualenv
```

mamba 默默地就把之前安装的torch vision 不断升级, 把torch升级到2.0了. 



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
2. kernel size增大1, 那要在四周 padding 1 . 
3. r i 的shape是什么?chunk分了之后  [batch size, hiddensize] ,  15是hidden size *3 
4. GRU 每一层输出的shape是什么?   out = [batch , sequence , hidden size ] , hidden = [sequence x bidirected, batch size , hidden size ]
5. 还有什么是线性不变的?  linear. 什么是线性不变? 
6. confusion matrix是什么样的.
7. softmax后 和是 1 . bmm 相乘

非常喜欢考shape.

## project4

### Part1 VAE

用Flickr-Faces-HQ Dataset.

##### 和 autoencoders 区别? (3 points)

与标准自动编码器相比，VAE 具有额外的概率组件，可以更有效地对潜在空间进行采样并生成新数据。

VAE 与标准自动编码器的不同之处在于，它们向编码器网络添加了一个概率元素，其中编码器的输出表示潜在空间的概率分布，而不是固定编码。

#### VAE怎么写?

we need to sample a latent vector from 正态分布,  using a simple reparametrization trick. This trick is important 为了梯度回传 to the encoder. 

最后用decoder产生图片根据given a latent vector. 值得注意的是VAEs still care about the mean squared error between the generated image and the real one (as in autoencoders), therefore the ideal output of a VAE is the average image over all plausible ones.

用frechet距离 between 两个高斯分布来评估. 第一个是数据集, 第二个是我们的样本. work on 提取出的特征. 

重建损失计算为 VAE 输出与原始输入图像之间的均方误差。正则化损失是使用学习概率分布和标准正态分布之间的 KL 散度计算的，这鼓励学习分布与先验分布相匹配。它通过正则化模型和降低潜在表示的复杂性来帮助防止过度拟合。

#### Task2

 why do we multiply beta by this factor? (check the loss formula)

 factor = latent_dim / image_size

因为 beta 参数控制正则化项的强度，并乘以一个取决于输入图像大小和潜在空间维数的因子。需要乘以该因子以确保正则化项相对于重建损失适当缩放。

用 frechet distance.计算两个多变量高斯分布的距离. 用多变量normal 来fit 高斯分布,也会给label fit一个高斯分布, 计算距离.

fid 256 的sota只有个位数, fid128数据集上训练 600多 距离.

#### task3

插值,扫描. 

weight变成 `[count, 1, 1]` tensor are broadcasted to match the dimensions of the `[N, F]` tensor as follows:  把NF 传播到 2 3 dim. 

能不能有更精细的控制? 传入条件, conditional variational auto encoder  CVAE

### part2 GAN 

实现一个DCGAN

#### Task1

图像是1x 32x32

不会显示建模数据分布. 没有把image encode到 latent vector.

用discriminator 辨别器,来指导decoder, 也就是生成器. 

D(x) 是一个二分图分类, 判断真还是假. 把生成的作为假样本. 

#### 生成器

1. 顶部分支的输入是什么（大小为100的向量）？ 顶部分支的输入是一个100维的均匀分布Z，通常被称为 "潜在向量"。这个向量作为生成器的随机性来源，它被用来合成一个可以骗过判别器的图像。

2. 底部输入大小为num_classes的单热编码标签向量。这个向量提供了一个条件性标签，指导生成器的生成过程。条件性GANs背后的想法是将生成器的条件放在一些特定的类信息上，**生成属于某个特定类的图像**。

3. 嵌入层将one hot编码的标签向量映射为特定大小的密集连续向量。这一层的目的是为每个类别学习一个有意义的嵌入，这可以帮助生成器更好地理解标签的基本语义。换句话说，嵌入层帮助生成器将特定的特征与特定的类联系起来。
4. transpose conv 通过将每层的特征图的高度和宽度增加一倍，生成器能够产生更高分辨率的图像。

transpose conv 是一种可用于对张量进行上采样的卷积运算。它与常规卷积运算相反. 

在 GAN 生成器的上下文中，转置卷积用于逐渐增加特征图的空间分辨率。特别是，生成器中的转置卷积将形状为张量作为输入`[batch_size, channels, height, width]`并产生`[batch_size, channels/2, height*2, width*2]`。通过将每一层特征图的高度和宽度加倍，生成器能够生成更高分辨率的图像。

conv 的图像变小:  $ \frac{n-f+2p}{s} +1https://pytorchbook.cn/chapter2/2.4-cnn/

#### task2

```
---> 37         x = torch.cat([x1, x2], dim=1)
     38         x = self.convs2(x)
     39         return x
RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 7 but got size 8 for tensor number 1 in the list.要resize 到32  x32
```

 BCE内部有sigmoid.

但是latent vector不是BatchSize , 100  ,1 , 1的形状吗? 为什么能变过去 4 x 512 x512? 为啥unflatten 就可以。 

#### 分辨器

1. what is the usage of the discriminator? 

指导generator 产生更接近实际分布的sample.

2. what is the output of the discriminator?

binary image classifer with a single scalar output classifying whether an input image is real or fake (generated).

#### task3网络inversion

GAN没有encoder, 把generator反过来 产生latent vector. 就是一个随机的z 梯度下降.对于image 计算reconstruction loss MSE. 

#### bonus对抗攻击

增加perturabations 扰动, 但是要人类难以察觉. 

Fast Sign Gradient Method (FGSM)

Untargeted, 最大化loss .

bird, 会变成frog ,攻击比较难. frog可以攻击成功. trunk可以攻击成bird.

#### 问题

6.5分.   问VAE原理， GAN的原理。encoder干嘛的， decoder干嘛的。

插值是怎么插入的? 

encoder 输出是啥, decoder输出是啥. 

