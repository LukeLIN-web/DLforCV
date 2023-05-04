3D Deep Learning Question: Read the PointNet (https://arxiv.org/pdf/1612.00593) and DGCNN paper https://arxiv.org/pdf/1801.07829 to answer the following questions: PointNet 

Questions: 

How does PointNet handle the problem of permutation invariance in point cloud data, and what are the advantages of this approach? 

PointNet use a simple symmetric function to aggregate the information of each point. After the MLP, they use max pooling. 

The advantages of this approach are several. First, it allows PointNet to handle unordered point clouds of varying sizes and densities, which is important for many real-world applications. Secondly, it can scale to big input elements.  Third, its accuracy is the higher than following two methods. 

1. ordering,  in high dimensional space there in fact does not exist an ordering that is stable w.r.t. point perturbations in the general sense
2. RNN, While RNN has relatively good robustness to input ordering for sequences with small length (dozens), it’s hard to scale to thousands of input elements, which is the common size for point sets.



Can you describe the architecture of PointNet and explain how it processes point cloud data? 

the architecture of PointNet has three key modules: the max pooling layer as a symmetric function to aggregate information from all the points, a local and global information combination structure, and two joint alignment networks that align both input points and point features.





How does the max pooling operation in PointNet help to aggregate information from multiple points, and how does it differ from other pooling operations in deep learning? 













DGCNN Questions:

1. How does DGCNN handle the problem of non-uniform sampling in 3D point clouds, and what are the advantages of this approach? 
2. Can you describe the dynamic graph construction process in DGCNN, and how it captures local and global spatial relationships between points? 
3. How does the dynamic graph pooling operation in DGCNN work, and how does it help to reduce the computational cost of processing large point clouds?











## 中文版

Questions: 

How does PointNet handle the problem of permutation invariance in point cloud data, and what are the advantages of this approach? 

Can you describe the architecture of PointNet and explain how it processes point cloud data? 

How does the max pooling operation in PointNet help to aggregate information from multiple points, and how does it differ from other pooling operations in deep learning? 

不变性: 

1. 直接将点云中的点以某种顺序输入（比如按照坐标轴从小到大这样） ->  很难找到一种稳定的排序方法
2. 作为序列去训练一个RNN，即使这个序列是随机排布的，RNN也有能力学习到排布不变性。 - > RNN很难处理好成千上万长度的这种输入元素（比如点云）。
3. 他们的方法:  使用一个简单的对称函数去聚集每个点的信息.  MLP 之后, 用max pooling. 他们的accuracy最高. 



架构:  input transform +  MLP +  feature transform + MLP + max pool + MLP 输出 output scores

**alignment network** 直接预测一个变换矩阵（3*3）来处理输入点的坐标。因为会有数据增强的操作存在，这样做可以在一定程度上保证网络可以学习到变换无关性。实际上，前半部分就是通过卷积和max_pooling对batch内各个点云提取global feature，再将global feature降到 3×K 维度，并reshape成 3x3 transform matrix,  通过数据增强丰富训练数据集，网络确实应该学习到有效的transform matrix，用来实现transformation invariance.

因为max pooling具有排列不变性. 





DGCNN Questions:

1. How does DGCNN handle the problem of non-uniform sampling in 3D point clouds, and what are the advantages of this approach? 
2. Can you describe the dynamic graph construction process in DGCNN, and how it captures local and global spatial relationships between points? 
3. How does the dynamic graph pooling operation in DGCNN work, and how does it help to reduce the computational cost of processing large point clouds?





2**EdgeConv**在网络的每一层上动态构建图结构，将每一点作为中心点来表征其与各个邻点的edge feature，再将这些特征聚合从而获得该点的新表征。

**EdgeConv**实现的实际就是通过构建局部邻域（这种局部邻域既可以建立在坐标空间，也可以建立在特征空间），对每个点表征。
