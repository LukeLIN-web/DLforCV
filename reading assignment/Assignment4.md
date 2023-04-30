# DLforCV

1 ViT

Read the most influential vision transformer paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (https://arxiv.org/abs/2010.11929v2). 

2 DETR 

Transformer-like architecture can be applied not only in the image classification task, but also in the fine-grained object detection task. Read DETR paper titled "End-to-End Object Detection with Transformers" (https://arxiv.org/abs/2005.12872). 

Question: 

1. Explain how transformers could be used for object detection. 

2. Can you briefly discuss what the object queries are in the DETR’s decoder?

#### detr

**在设计损失函数的地方，采用匈牙利算法，为网络生成的固定数量的预测框，唯一的分配最优的gt框**，然后去计算对应的loss。**并且本文的transformer模块对于输入的长度为N的object queries，是并行解码的**

首先通过CNN提取特征，随后对特征图进行降维然后展平之后，与位置编码信息叠加在一起，送入[transformer](https://www.zhihu.com/search?q=transformer&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"438770010"})结构中，随后输出一组预测结果，像上图中每个带有颜色的小方块都是一个预测结果，结果包含有类别和预测框。本文将得到的检测结果和GroundTruth进行**二部图匹配**计算loss。简单点来说就是本文固定输出为N=100个预测目标，然后与之相应的Groundtruth也会使用空标记来进行补齐，达到100个。然后双方数目一样后，就可以用匈牙利算法进行二部图匹配，就是将预测框逐个分配给最优匹配的groundtruth框。



#### Explain how transformers could be used for object detection. 

Transformers can be used for object detection by incorporating them into an object detection pipeline. One way to do this is to use transformers to generate features from the image, which can then be used to identify objects in the image using an object detection algorithm such as YOLO or Faster R-CNN.

To use transformers for feature extraction in object detection, we can first break the image up into patches and feed them through a pre-trained transformer model. The resulting output features can then be combined into a feature map, which can be fed into an object detection algorithm to identify objects and their locations within the image.

The input of the **transformer encoder** is a sequence, Each encoder layer consists of a multi-head self-attention module and a FFN. Since the encoder is permutation-invariant, detr introduces position encoding, which is added to the input of all attention layers.

Detr can decode in parallel, while previous transformers decode sequentially. Since the decoder is also permutation-invariant, the N input embeddings must be different to produce different results. These input embeddings are learnt positional encodings that we refer to as object queries, and similarly to the encoder, we add them to the input of each attention layer. The N object queries are transformed into an output embedding by the decoder. They are then independently decoded into box coordinates and class labels by a feed forward network, resulting N final predictions. Using self- and encoder-decoder attention over these embeddings, the model globally reasons about all objects together using pair-wise relations between them, while being able to use the whole image as context.

#### Can you briefly discuss what the object queries are in the DETR’s decoder?

Object queries are positional embeddings, where each position corresponds to a predicted result after passing through the decoder. Object queries are combined with the image's feature information and passed through the decoder layers to obtain the predicted results.

N object queries are transformed into output embeddings through the decoder, and then independently encoded into box coordinates and class labels through a feed-forward network, resulting in N predicted results.

Object queries can be seen as asking the model what is in a certain location in the image, so when the number of object queries is set large enough, they can sufficiently cover the entire image, thereby avoiding the issue of missing predictions.



#### ViT

由于NLP处理的语言数据是序列化的，而CV中处理的图像数据是三维的，所以我们需要一个方式将图像这种三维数据转化为序列化的数据。

如何将2D图片数据转换成 1D数据？目前 BERT 能够处理的序列长度是512，如果直接将图像像素转换成 1D。即使是 224 × 224 大小的图片，其序列长度也有5万多，计算复杂度将是 BERT 的100倍，如果是检测或分割任务，那计算复杂度就更大了。

所以将自注意力机制应用在CV领域，关键在于将图片分割等大的patch，添加位置信息，然后按序排成一排，输入进Vision Transformer进行训练

为了模型不受patch大小的影响，作者引入了[线性映射](https://www.zhihu.com/search?q=线性映射&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"477168874"})来把每个patch大小线性映射成固定的D维。

由于transformer模型本身是没有位置信息的，和NLP中一样，我们需要用位置嵌入将位置信息加到模型中去。



#### 解释如何将变压器用于物体检测。

变换器可以通过将其纳入物体检测管道来用于物体检测。一种方法是使用变换器从图像中生成特征，然后可以使用YOLO或Faster R-CNN等物体检测算法来识别图像中的物体。

为了在物体检测中使用变换器进行特征提取，我们可以首先将图像分解成补丁，并将其送入一个预先训练好的变换器模型。然后，所产生的输出特征可以组合成一个特征图，并将其送入物体检测算法，以识别物体及其在图像中的位置。

变换器编码器**的输入是一个序列，每个编码器层由一个多头自注意模块和一个FFN组成。由于编码器是互变的，Detr引入了位置编码，它被添加到所有注意力层的输入中。

Detr可以并行解码，而之前的变换器是按顺序解码的。由于解码器也是变异的，N个输入嵌入必须是不同的，以产生不同的结果。这些输入嵌入是学习到的位置编码，我们称之为对象查询，与编码器类似，我们将它们添加到每个注意力层的输入中。N个对象查询被解码器转化为输出嵌入。然后，它们被一个前馈网络独立解码为箱体坐标和类别标签，产生N个最终预测。在这些嵌入上使用自我和编码器-解码器注意力，该模型使用它们之间的成对关系对所有物体进行全局推理，同时能够使用整个图像作为背景。

#### 你能简单地讨论一下DETR解码器中的物体查询是什么吗？

对象查询是位置嵌入，在通过解码器后，每个位置对应一个预测的结果。对象查询与图像的特征信息相结合，并通过解码器层来获得预测的结果。

N个物体查询通过解码器转化为输出嵌入，然后通过前馈网络独立编码为箱体坐标和类别标签，从而得到N个预测结果。

物体查询可以看作是向模型询问图像中的某个位置有什么，所以当物体查询的数量设置得足够大时，它们可以充分覆盖整个图像，从而避免了预测结果缺失的问题。
