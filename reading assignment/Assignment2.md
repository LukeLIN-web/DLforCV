# DLforCV

Briefly discuss the contributions of each of the two papers above. 

Compare segmentation and object detection tasks in general and then highlight the difference between the architectures and tricks used for object detection to  those(architectures and tricks) used in image segmentation.

## Contribution

#### YOLO

*We present YOLO, a new approach to object detection.*

Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a re gression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance.

1. Our unified architecture is extremely fast. 
2. 

Compared to SOTA detection systems, YOLO makes more localization errors but is less likely to predict false positives on background. Finally, YOLO learns very general representations of objects. It outperforms other detection methods, including DPM and R-CNN, when gener- alizing from natural images to other domains like artwork.

YOLO works across a variety of natural images. also generalizes well to new domains , like art.

YOLO outperforms methods like DPM and R-CNN when generalizing to person detection in artwork.

#### DeepLabV3









## 不同

highlight the difference between the architectures and tricks used for object detection to those used in image segmentation. 

#### yolo

##### architectures

before yolo, they use  sliding windows, yolo split the image into a grid, each cell predicts boxes and confidences.   then we combine the box and class predictions.

Finally we do nms and threshold detections.   

This parameterization fixed output, thus we can train NN to be a whole detection pipeline. During training , match example to the right cell.   find the best one , adjust it, increase the confidence.  

decrease the confidence of other boxes. some cells don't have any ground truth detections.

Don't adjust the class probabilities or coordinates. 





#####  tricks





#### deeplabv3

##### architectures

扩大的dilated convolutions.  Parallel modules with atrous convolution (ASPP), augmented with image-level features.  has a  Atrous Spatial Pyramid Pooling.

 figure5 





#####  tricks
