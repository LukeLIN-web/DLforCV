3D Deep Learning Question: Read the PointNet (https://arxiv.org/pdf/1612.00593) and DGCNN paper https://arxiv.org/pdf/1801.07829 to answer the following questions: PointNet 

Questions: 

## PointNet

### Question1

How does PointNet handle the problem of permutation invariance in point cloud data, and what are the advantages of this approach? 

PointNet use a symmetric function to aggregate the information of each point. After the MLP, they use max pooling. 

The advantages:

First, it allows PointNet to handle unordered point clouds of varying sizes and densities, which is important for many real-world applications. 

Secondly, it can scale to big input elements.

Third, its accuracy is higher than following two methods:

1. ordering,  in high dimensional space there in fact does not exist an ordering that is stable w.r.t. point perturbations in the general sense
2. RNN, While RNN has relatively good robustness to input ordering for sequences with small length (dozens), it’s hard to scale to thousands of input elements, which is the common size for point sets.

### Question2

Can you describe the architecture of PointNet and explain how it processes point cloud data? 

architecture :  input transform + shared MLP +  feature transform + shared MLP + max pool + MLP +  output scores

the architecture of PointNet has three key modules: the max pooling layer as a symmetric function to aggregate information from all the points, a local and global information combination structure, and two joint alignment networks that align both input points and point features.

The first step is input and feature transform, it uses a T-Net and matrix multiply.

The second step is to apply a shared Multi-Layer Perceptron (MLP) to each point independently. This MLP consists of several fully connected layers with ReLU activation functions, which allows the network to learn non-linear mappings from the input space to feature space. 

Next, PointNet aggregates information across all points using a max pooling. This computes the maximum value for each feature across all points in the set, resulting in a fixed-length global feature vector that summarizes the entire point cloud.

Finally, this global feature vector is fed through another MLP to produce either class labels for the entire input or per-point segment/part labels for each point of the input.

### Question3

How does the max pooling operation in PointNet help to aggregate information from multiple points, and how does it differ from other pooling operations in deep learning? 

Max pooling works by taking the maximum value of each feature across all points in the set. This results in a fixed-length global feature vector that summarizes the entire point cloud. By taking the maximum value, max pooling effectively selects **the most informative features from each point** and aggregates them into a single vector. 

Compared to other pooling operations in deep learning, such as average pooling or sum pooling, max pooling has several advantages. 

First, it is more robust to noise and outliers because it only considers the maximum value rather than averaging or summing all values. 

Second, it preserves spatial information because it only takes the maximum value within each local region rather than aggregating all values together. 

Finally, max pooling is computationally efficient because it can be implemented using simple element-wise comparisons.  In Figure 5, max- pooling operation achieves the best performance by a large winning margin.

## DGCNN Questions

1. How does DGCNN handle the problem of non-uniform sampling in 3D point clouds, and what are the advantages of this approach? 
2. Can you describe the dynamic graph construction process in DGCNN, and how it captures local and global spatial relationships between points? 
3. How does the dynamic graph pooling operation in DGCNN work, and how does it help to reduce the computational cost of processing large point clouds?

### q1

DGCNN addresses this problem by **introducing a dynamic graph convolutional layer** that constructs a graph representation of the point cloud, where each point is represented as a node in the graph and the edges between nodes are determined based on the proximity of the points in the 3D space. 

This allows for adaptive sampling of the points, since the graph edges are determined dynamically based on the local point density.

The dynamic graph convolutional layer then performs convolution on the graph to extract features from the local neighborhood of each point. The resulting feature vectors are then concatenated with the original point features, and passed through a series of MLP layers for further processing.

Advantages: 

First, it allows for **adaptive sampling** of the points, which ensures that all regions of the object surface are sampled with sufficient density. 

Second, the graph convolutional layer allows for efficient feature extraction from the local neighborhood of each point, which can capture fine-grained geometric details. 

Finally, the use of MLP layers after the graph convolutional layer enables the network to learn high-level features and make accurate predictions for various 3D point cloud tasks, such as classification and segmentation.

### q2

EdgeConv dynamically constructs a graph structure at each layer of the network, using each point as the center point to represent its edge features with neighboring points, and then aggregates these features to obtain a new representation for the point.

EdgeConv actually implements the method of representing each point by constructing a local neighborhood (which can be established in both coordinate space and feature space).

Step 1: Use the EdgeConv module to represent the point-wise features of the input data (including coordinate space and feature space) layer by layer, with the output of the previous EdgeConv module being the input of the next EdgeConv module. 

Q: How it captures local and global spatial relationships between points? 

Answer : An dedicated aggregation method is selected, using an asymmetric edge function that explicitly combines global shape structure captured by the coordinates of the patch centers xi with local neighborhood information captured by xj − xi.

Step 2: The point-wise features at different levels are then concatenated together, and global features are obtained through max pooling.

### q3

As we said in question2, step 2 uses a global max pooling  to get the point cloud global feature.

The dynamic graph pooling operation helps to reduce the computational cost of processing large point clouds by downsampling the data, reducing its size and complexity. This allows for faster processing times and reduces the memory requirements of the model. 



## 中文版

Questions: 

How does PointNet handle the problem of permutation invariance in point cloud data, and what are the advantages of this approach? 

Can you describe the architecture of PointNet and explain how it processes point cloud data? 

How does the max pooling operation in PointNet help to aggregate information from multiple points, and how does it differ from other pooling operations in deep learning? 

不变性: 

1. 直接将点云中的点以某种顺序输入（比如按照坐标轴从小到大这样） ->  很难找到一种稳定的排序方法
2. 作为序列去训练一个RNN，即使这个序列是随机排布的，RNN也有能力学习到排布不变性。 - > RNN很难处理好成千上万长度的输入元素（比如点云）。
3. 他们的方法:  使用一个简单的对称函数去聚集每个点的信息.  MLP 之后, 用max pooling. 他们的accuracy最高.  faster, stable. deal with long length input.

架构:  input transform +  MLP +  feature transform + MLP + max pool + MLP 输出 output scores

**alignment network** 直接预测一个变换矩阵（3*3）来处理输入点的坐标。因为会有数据增强的操作存在，这样做可以在一定程度上保证网络可以学习到变换无关性。实际上，前半部分就是通过卷积和max_pooling对batch内各个点云提取global feature，再将global feature降到 3×K 维度，并reshape成 3x3 transform matrix,  通过数据增强丰富训练数据集，网络确实应该学习到有效的transform matrix，用来实现transformation invariance.

因为max pooling具有排列不变性. more robust to noise and outliers, faster 

### DGCNN Questions:

1. How does DGCNN handle the problem of non-uniform sampling in 3D point clouds, and what are the advantages of this approach? 
2. Can you describe the dynamic graph construction process in DGCNN, and how it captures local and global spatial relationships between points? 
3. How does the dynamic graph pooling operation in DGCNN work, and how does it help to reduce the computational cost of processing large point clouds?

### 问题1 

adaptive sampling 

efficient feature extraction from the local neighborhood of each point

MLP可以学到 high-level features 

#### 问题2

EdgeConv在网络的每一层上动态构建图结构，将每一点作为中心点来表征其与各个邻点的edge feature，再将这些特征聚合从而获得该点的新表征。

**EdgeConv**实现的实际就是通过构建局部邻域（这种局部邻域既可以建立在坐标空间，也可以建立在特征空间），对每个点表征。

step1 :逐级对输入（包括坐标空间和特征空间）使用EdgeConv模块表征point-wise feature，也就是前一个EdgeConv模块的输出又为下一个EdgeConv模块的输入。

具体实现 :  使用了4个EdgeConv首尾依次相接。 

聚合方法   an asymmetric edge function, This explicitly combines global shape structure, captured by the coordinates of the patch centers xi , with local neighborhood infor- mation, captured by xj −xi . 

step2 : 接着将不同层次的[point-wise feature](https://www.zhihu.com/search?q=point-wise feature&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A"267895014"})拼接起来，通过max pooling得到global feature 

#### 问题3

As we said in question2, step 2 uses a global max pooling  to get the point cloud global feature.

The dynamic graph pooling operation helps to reduce the computational cost of processing large point clouds by downsampling the data, reducing its size and complexity. This allows for faster processing times and reduces the memory requirements of the model. 
