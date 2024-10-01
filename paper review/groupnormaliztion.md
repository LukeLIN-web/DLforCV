Read and write a review for [Group Normalization]

#### Reviewer

 Juyi Lin , 09/28/2024

#### Citation

**Yuxin Wu, Kaiming He**; Group Normalization. Proceedings of the European Conference on Computer Vision (ECCV), 2018, pp. 3-19. https://doi.org/10.48550/arXiv.1803.08494

#### Brief summary of the paper

There's a difference between the mean and variance during training and testing, , which causes Batch Normalization's error to increase rapidly as batch size decreases. This paper introduces Group Normalization(GN) as a simple alternative to batch Normalization(BN). GN divides the channels into groups and computes within each group the mean and variance for normalization. GN can effectively replace the powerful BN in a variety of tasks.

#### Main contribution

1. GN has obvious lower error rate than BN when batch size is small; 
2. GN is positioned as an alternative to BN, Layer Normalization (LN), and Instance Normalization (IN), and it demonstrates better performance in many computer vision tasks with small batch sizes.

#### Strengths

1. Superior performance in cases where the batch size is small (= 2), with GroupNorm showing 10% lower error compared to BatchNorm.
2. GN can be easily implemented by a few lines of code.
3. Versatility Across Tasks 
4. Stability Across Batch Sizes.

#### Weakness

1. GN loses some regularization ability of BN.

#### More detailed explanation of the strengths 

1. GN is shown to be effective beyond just image classification; the paper demonstrates its strong performance in object detection (Mask R-CNN on COCO), segmentation, and video classification (Kinetics), highlighting its versatility in various applications.
2. GN maintains stable performance across a wide range of batch sizes, allowing for greater flexibility
3. GN can be implemented with minimal complexity, facilitating adoption in existing frameworks and models without significant retraining of architectures.

#### More detailed explanation of the weakness

1. BN’s mean and variance computation introduces uncertainty caused by the stochastic batch sampling, which helps regularization. This uncertainty is missing in GN.

#### Comments about the experiments

Are they convincing? 

1. Figure 4: GN is compared with LN, BN, and IN on ResNet-50. The drawback is that experiments were not conducted on more datasets. Figure 5: It compares BN and GN across different batch sizes, demonstrating that when the batch size is less than or equal to 8, the validation error is lower; however, experiments were not conducted with LN and IN.
2. Table 3: They explored the optimal number of groups for achieving the lowest validation error. This experiment was compared with IN and LN.  Tables 4-6: They compare different backbones, with multiple experiments demonstrating that GN achieves better accuracy in object detection and segmentation. In Table 7, they train the models from scratch.
3. The authors evaluate Group Normalization (GN) across various tasks, including image classification (ImageNet), object detection and segmentation (COCO) , and video classification (Kinetics) Figure7. This comprehensive approach demonstrates GN’s versatility and effectiveness in different contexts.
4. GN demonstrates stable performance across a wide range of batch sizes, which is essential for practical applications where memory constraints often limit batch size.

#### How could the work be extended?

1. GN loses some regularization ability of BN. It is possible that GN combined with a suitable regularizer will improve results. This can be a future research topic.
2. Comparison with Other Normalization Methods: While the paper primarily focuses on Group Normalization (GN) and Batch Normalization (BN), future research could investigate how GN compares with other normalization techniques, such as Layer Normalization (LN) and Instance Normalization (IN), across different types of neural network architectures and tasks beyond those evaluated in this study.
3. Many state-of-the-art systems have been designed with BN. Future work could involve redesigning these systems or conducting hyperparameter searches specifically for GN to potentially yield better performance.

#### Additional comments

**Unclear points**: In practice, there are not only two modes, train and test; the specifics of how to perform fine-tuning remain unclear. What is the effect of normalization when fine-tuning a GN pretrained model?

**Open research questions**: Since GN addresses the batch size limitation of BN, an extension could involve training larger models or more complex architectures with GN and evaluating performance gains.

**Applications**: Group Normalization (GN) is particularly effective in most computer vision tasks with small batch sizes, such as object detection, image segmentation, and video classification.