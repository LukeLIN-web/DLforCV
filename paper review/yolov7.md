YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

#### Reviewer

 Juyi Lin, 10/02/2024

#### Citation

Wang, Chien-Yao, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*. 2023. https://doi.org/10.48550/arXiv.2207.02696

#### Brief summary of the paper

This paper introduces an enhanced version of the YOLO (You Only Look Once) series, designed for real-time object detection. YOLOv7 incorporates several key innovations, such as **trainable bag-of-freebies** techniques, efficient architectural modifications, and dynamic label assignment, all aimed at improving accuracy without increasing inference costs. The model achieves state-of-the-art performance on standard datasets like COCO, surpassing previous YOLO models and real-time detectors in both speed (FPS) and accuracy. YOLOv7 strikes an optimal balance between speed and precision, making it highly efficient for real-time applications while maintaining low-latency and high accuracy.

#### Main contribution

1. Several different trainable bag-of-freebies methods were designed, which can improve the accuracy of the network without increasing inference time. 
2. The paper discovered how to re-parameterize and addressed the dynamic label assignment problem in multi-branch networks, proposing a solution. 
3. It also introduced extend and compound scaling methods, which efficiently utilize parameters and computation.

#### Strengths

1. Better Accuracy without Extra Inference Cost
2. Optimal Speed-Accuracy 
3. Efficient Architecture

#### Weakness

1. Experimental Setup and Generalizability
2. Complexity of Implementation
3. Lack of Comparative Analysis

#### More detailed explanation of the strengths 

1. The introduction of the “trainable bag-of-freebies” methods significantly enhances detection accuracy without increasing inference costs, effectively improving the model’s performance through efficient training tools.
2. It outperforms previous YOLO models and other real-time detectors like EfficientDet and YOLOv5 by offering: Higher FPS (frames per second) and SOTA accuracy. Faster inference speed due to efficient architectural modifications and optimizations. Plus, it maintains the low-latency required for real-time applications.
3. YOLOv7 introduces architectural improvement: **Extended Efficient Layer Aggregation Networks (ELAN)**: This design improves feature propagation and representation while keeping computational cost low, enhancing the model’s ability to extract better features from input images.

#### More detailed explanation of the weakness

1. While the experiments are conducted using the MS COCO dataset, the findings may not necessarily generalize to other datasets or real-world applications without further validation.
2. The introduction of advanced methods like planned re-parameterization and dynamic label assignment may increase the complexity of the implementation, which could be a barrier for practitioners looking to adopt the methods.
3. Although the paper compares YOLOv7 with other models, it may not provide an in-depth analysis of certain established models that also focus on optimization or inference speed, which could give a more rounded view of the state of the art.

#### Comments about the experiments

Are they convincing? 

1. Figure 1 compares the inference speed and accuracy of several detectors, and the chart is clear and straightforward.
2. Table 1 and Table 2 provide a detailed comparison of a series of previous YOLO frameworks, with multiple accuracy metrics thoroughly compared.
3. The results are quite reliable since multiple ablation studies were conducted. Tables 3 to 8 show the ablation experiments, proving that each architectural optimization is useful.
4. However, the experiments were only conducted on the COCO dataset. It would be more convincing if tested on additional datasets.

#### How could the work be extended?

1. It can be fine-tuned on different types of datasets.
2. New architectures could be developed to better generalize across a broader range of object detection tasks, including those with limited or noisy data.

#### Additional comments

**Unclear points**: No Consideration of Adverse Effects: The paper does not discuss any potential drawbacks or adverse effects of implementing the proposed methods, such as the possibility of overfitting or the difficulties encountered when deploying the models in real-time scenarios.

**Open research questions**: YOLOv7’s performance is outstanding on standard datasets like COCO, but generalization across domains remains a challenge. How can the model be made more adaptable to new or unseen environments like autonomous vehicles without extensive re-training?

**Applications**: YOLOv7 can be applied in various fields, including autonomous driving, surveillance, robotics, and any real-time application requiring efficient object detection.