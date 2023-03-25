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





