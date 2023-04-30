# DLforCV

Briefly discuss the contributions of each of the two papers above. 

Compare segmentation and object detection tasks in general 

and then highlight the difference between the architectures and tricks 

used for object detection to  those(architectures and tricks) used in image segmentation.

## Contribution

#### YOLO

1. Unified architecture is extremely fast. 
2. Compared to SOTA detection systems, YOLO makes more localization errors but is less likely to predict false positives in the background. Finally, YOLO learns very general representations of objects. It outperforms other detection methods, including DPM and R-CNN, when generalizing from natural images to other domains like artwork.
3. YOLO works across a variety of natural images. It also generalizes well to new domains, like art. YOLO outperforms methods like DPM and R-CNN when generalizing to person detection in the artwork.

#### DeepLabV3

1. DeepLabV3 introduced an atrous convolutional neural network (ACNN) architecture that can capture multi-scale contextual information at different resolutions without increasing the number of parameters.
2. DeepLabV3 improved the accuracy of semantic segmentation by using a pyramid pooling module that captures feature maps at multiple scales and aggregates them using global pooling.
3. DeepLabV3 demonstrated SOTA performance on several semantic segmentation benchmarks, including the PASCAL VOC and Cityscapes datasets.

## general comparsion

### object detection

Object detection involves identifying and localizing target objects within an image. This task not only provides information about the class of the object, but also its coordinates. Even if multiple objects are present in the image, the network can detect them individually. Object detection does not necessarily focus on the boundaries of the objects.

### segmentation

Image segmentation involves extracting objects from an image and determining which pixels belong to each object. This task aims to obtain a new image by assigning each pixel to a specific object. Image segmentation pays particular attention to the boundaries of objects to accurately separate them from the background.

Semantic segmentation is no objects, just pixels.

Instance segmentation has multiple objects.

## Architecture

### object detection

 **Region Proposal**: First, a set of region proposals is generated using a selective search algorithm. These region proposals are potential object bounding boxes in the image. 

**Feature Extraction**: Each region proposal is then passed through a pre-trained CNN to extract a feature vector. The CNN is typically pre-trained on the ImageNet dataset. 

**Object Detection**: The feature vectors are then used to train a set of linear SVMs (Support Vector Machines) to classify the region proposals into object categories or background.  

**Bounding Box Regression**: Finally, the output of the SVMs is passed through a regression model that refines the location of the bounding boxes.

faster RCNN  is a 2 stage object detector. 

1. run once per image.  back bone , region proposal 
2. run once per region.  crop feature, predict object class, prediction bbox offset.

### segmentation

The architecture consists of a feature extractor, an encoder-decoder network, and a classifier. 

The feature extractor is typically a pre-trained CNN such as ResNet, which extracts high-level features from the input image. 

The encoder-decoder network is used to refine the feature maps and produce a more accurate segmentation map. The encoder typically consists of multiple convolutional layers, while the decoder consists of up-sampling layers followed by convolutional layers.

### Difference

#### backbone

Object detection models typically use backbone architectures such as ResNet or MobileNet as feature extractors to generate a high-level feature representation of the input image. These features are then used to predict object locations and labels. 

Image segmentation models often use encoder-decoder architectures such as U-Net, DeepLabv3, or Mask R-CNN to extract high-level features from the input image and then upsample them to generate a dense pixel-wise output.

#### Loss functions

Object detection models typically use a combination of classification and localization loss functions such as cross-entropy and smooth L1, respectively. Image segmentation models often use pixel-wise loss functions such as cross-entropy, Dice coefficient, or Jaccard index to optimize the pixel-wise segmentation accuracy.

#### Multi-scale feature extraction:

Both object detection and image segmentation models benefit from extracting features at multiple scales. However, the specific techniques used for multi-scale feature extraction can differ. Object detection models often use feature pyramids or feature maps at multiple resolutions to detect objects of different sizes and scales. In image segmentation, encoder-decoder architectures like U-Net, DeepLabv3, or Mask R-CNN are used to extract features at different resolutions, which are then combined to generate a high-resolution segmentation map.

## Tricks

### object detection

Prior work on object detection repurposes classifiers to perform detection. Fast R-CNN, insert region proposal network to predict proposals from features. Crop features for each proposal, and classify each one.

YOLO: split the image into a grid. Each cell predicts boxes and confidences. Yolo combines the box and class predictions. Finally, we do nms and threshold detections. Yolo frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network directly predicts bounding boxes and class probabilities from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

Yolo can train NN to be a whole detection pipeline. During training, match the example to the correct cell. Find the best one, adjust it, and increase the confidence. Decrease the confidence of other boxes. Some cells don't have any ground truth detections.

### segmentation

object detection generates a set of candidate bounding boxes that may contain objects of interest. Techniques like selective search, region proposal networks (RPN), and anchor-based methods like Faster R-CNN are commonly used for this purpose. In contrast, image segmentation models often use pixel-wise prediction directly without any proposal stage.

Atrous Spatial Pyramid Pooling (ASPP) module that uses parallel modules with atrous convolution to capture multi-scale information at different levels of resolution. This enables the network to capture context at different scales and resolutions, essential for accurate segmentation.

In addition to the ASPP module, DeepLabv3 incorporates image-level features to refine the segmentation results further. The encoder network uses atrous convolutions to capture multi-scale contextual information, while the decoder network uses bilinear interpolation to up-sample the feature maps and recover the spatial information lost during the encoding process.
Overall, the combination of dilated convolutions and spatial pyramid pooling in DeepLabv3 enables the network to capture local and global contextual information effectively.



## 贡献

#### YOLO

1. 统一的架构是非常快的。
2. 与SOTA检测系统相比，YOLO的定位错误较多，但在背景中预测假阳性的可能性较小。最后，YOLO学习了非常普遍的物体表征。当从自然图像泛化到艺术品等其他领域时，它优于其他检测方法，包括DPM和R-CNN。
3. YOLO适用于各种自然图像。它还能很好地概括到新的领域，如艺术品。当泛化到艺术品中的人物检测时，YOLO的性能优于DPM和R-CNN等方法。

#### DeepLabV3

1. DeepLabV3引入了一个反卷积神经网络（ACNN）架构，可以在不增加参数数量的情况下捕捉不同分辨率的多尺度背景信息。
2. 2.DeepLabV3通过使用金字塔池化模块提高了语义分割的准确性，该模块在多个尺度上捕获特征图，并使用全局池化来聚合它们。
3. DeepLabV3在几个语义分割基准上展示了SOTA性能，包括PASCAL VOC和Cityscapes数据集。

## 一般比较

### 对象检测

对象检测包括识别和定位图像中的目标对象。这项任务不仅提供关于物体类别的信息，而且还提供其坐标。即使图像中存在多个物体，网络也能单独检测它们。对象检测不一定关注对象的边界。

###分割

图像分割包括从图像中提取物体并确定哪些像素属于每个物体。这项任务的目的是通过将每个像素分配给一个特定的物体来获得一个新的图像。图像分割特别注意物体的边界，以准确地将它们与背景分开。

语义分割没有对象，只有像素。

实例分割有多个对象。

## 架构

对象检测

 **区域提议**： 首先，使用选择性搜索算法生成一组区域建议。这些区域建议是图像中潜在的物体边界盒。

**特征提取**： 每个区域建议然后通过一个预先训练的CNN来提取特征向量。该CNN通常在ImageNet数据集上进行预训练。

**物体检测**： 然后，特征向量被用来训练一组线性SVM（支持向量机），将区域建议分类为物体类别或背景。 

**边界箱回归**： 最后，SVM的输出被传递到一个回归模型中，以细化边界框的位置。

更快的RCNN是一个2阶段的物体检测器。

1.每幅图像运行一次，背骨，区域建议 
2. 每个区域运行一次。 裁剪特征，预测物体类别，预测bbox offset。

### 分割

该架构由一个特征提取器、一个编码器-解码器网络和一个分类器组成。

特征提取器通常是一个预先训练好的CNN，如ResNet，它从输入图像中提取高级特征。

编码器-解码器网络用于细化特征图，并产生更准确的分割图。编码器通常由多个卷积层组成，而解码器则由上采样层和卷积层组成。

### 差异

#### 骨干网

物体检测模型通常使用骨干架构，如ResNet或MobileNet作为特征提取器，生成输入图像的高级特征表示。然后，这些特征被用来预测物体位置和标签。

图像分割模型通常使用编码器-解码器架构，如U-Net、DeepLabv3或Mask R-CNN，从输入图像中提取高级特征，然后对其进行升采样，生成密集的像素级输出。

#### 损失函数

物体检测模型通常使用分类和定位损失函数的组合，如交叉熵和平滑L1，分别。图像分割模型通常使用像素级的损失函数，如交叉熵、Dice系数或Jaccard指数来优化像素级的分割精度。

#### 多尺度特征提取：

物体检测和图像分割模型都得益于多尺度特征的提取。然而，用于多尺度特征提取的具体技术可能有所不同。物体检测模型通常使用多分辨率的特征金字塔或特征图来检测不同尺寸和尺度的物体。在图像分割中，像U-Net、DeepLabv3或Mask R-CNN这样的编码器-解码器架构被用来提取不同分辨率的特征，然后将其结合起来，生成高分辨率的分割图。

## 技巧

### 物体检测

之前关于物体检测的工作重新利用分类器来进行检测。快速R-CNN，插入区域提议网络，从特征中预测提议。为每个提议裁剪特征，并对每个提议进行分类。

YOLO：将图像分成一个网格。每个单元格预测方框和置信度。Yolo结合盒子和类别预测。最后，我们做nms和阈值检测。Yolo将物体检测作为一个回归问题，以空间上分离的边界盒和相关的类概率为框架。一个神经网络在一次评估中直接从完整的图像中预测边界盒和类别概率。由于整个检测管道是一个单一的网络，可以直接对检测性能进行端到端的优化。

Yolo 可将 NN 训练成整个检测管道。在训练过程中，将实例与正确的 Yolo 可将 NN 训练成整个检测管道。在训练过程中，将例子与正确的单元匹配。找到最好的一个，调整它，并增加信心。降低其他单元格的置信度。有些单元格没有任何基础事实的检测。

### 分割

对象检测产生一组可能包含感兴趣的对象的候选边界盒。像选择性搜索、区域建议网络（RPN）和基于锚点的方法（如Faster R-CNN）等技术通常用于此目的。相比之下，图像分割模型通常直接使用像素级的预测，而没有任何提议阶段。

Atrous Spatial Pyramid Pooling（ASPP）模块，该模块使用平行模块与Atrous卷积来捕捉不同分辨率水平的多尺度信息。这使网络能够在不同的尺度和分辨率下捕获上下文，这对准确的分割至关重要。

除了ASPP模块外，DeepLabv3还加入了图像级别的特征，以进一步完善分割结果。编码器网络使用扩张卷积来捕捉多尺度的上下文信息，而解码器网络使用双线性插值来对特征图进行上样，并恢复编码过程中丢失的空间信息。
总的来说，DeepLabv3中的扩张卷积和空间金字塔池的结合使网络能够有效地捕获局部和全局的上下文信息。

