Read and write a review for [ImageNetChallenge]

#### Reviewer

 Juyi Lin , 09/18/2024

#### Citation

Russakovsky, O., Deng, J., Su, H. *et al.* ImageNet Large Scale Visual Recognition Challenge. *Int J Comput Vis* **115**, 211–252 (2015). https://doi.org/10.1007/s11263-015-0816-y 

#### brief summary of the paper

This paper introduces a benchmark for object category classification and detection across hundreds of object categories and millions of images. It provides a detailed explanation of how the dataset was collected and annotated. The paper also discusses improvements made to the dataset over the past five years, introduces various outstanding algorithms that have emerged in the competition over the years, and proposes future directions and improvements.

#### Main contribution

1. It can take a closer look at the current state of the field of categorical object recognition.  						 					 				 			 		
2. perform an analysis of statistical properties of objects and their impact on recognition algorithms.
3. It provides a detailed explanation of how the dataset was collected and annotated. Therefore, it presents a comprehensive example of the massive dataset collection process.

#### strengths

1. It proposes a benchmark for object classification and detections solutions. It is wide range and comprehensive and detailed.
2. contains many more fine-grained classes compared to the previous benchmark (such as standard PASCAL VOC )
3. It explains how to quickly acquire multiple labels, such as relevance.
4. The dataset has been continuously improved, addressing several concerns over five years.

#### weakness

2. Imbalanced Classes, Certain object categories have far more samples than others.
3. incorrect labels
4. Overemphasis on Accuracy Metrics
5. Overfitting to the Competition
6. Benchmark Saturation

#### More detailed explanation of the strengths 

1. They found a way to efficiently perform multi-label annotation by accounting for the real-world structure of data, including label correlations, hierarchical organization of concepts, and label sparsity.
2. They conducted several rounds of post-processing on the annotations obtained via crowdsourcing, correcting many common sources of errors.

#### More detailed explanation of the weakness

1. It outlines the specific details of each year's *ILSVRC* competition. However, some parts are overly detailed; for example, tables 6-8 could be moved to the appendix.
2. Single-object localization task. minimum per class is 92 , maximum per class 1418,相差15.41倍. It can skew models to perform well on popular categories while underperforming on less-represented ones.
3. Some categories in ImageNet have noisy or incorrect labels. For example, objects in certain images are mislabeled or ambiguous, which can lead to noisy supervision during training and hinder performance.
4. The challenge focused on top-5 accuracy for classification, which doesn’t always reflect real-world needs. In practical applications, misclassifying an object might be more serious (e.g., self-driving cars misidentifying pedestrians), so accuracy metrics alone can be limiting.
5. Over the years, many algorithms were specifically optimized to do well on ImageNet (overfitting to the dataset's structure)
6. near-human accuracy levels, After models like ResNet reached performance close to human accuracy, the challenge started to lose relevance as a benchmark.

#### Comments about the experiments

Are they convincing?

Yes

1. Because a large number of third parties participated in reproducing the results.
2. They publicly released the Object Categories and the Hierarchy of Questions.

#### How could the work be extended?

1. A pixel-level object segmentation dataset could be developed.
2. The dataset could be generalized to images from different countries, or low-quality images from smartphones.
3. Other performance metrics can be considered, such as inference time, memory usage.**VGG** and **ResNet**, were extremely large and resource-intensive, making them impractical for deployment in real-time or resource-constrained environments like mobile devices.
4. Explainability. Understanding why a model made a particular decision is critical in many domains, such as healthcare or autonomous vehicles.

#### Additional comments

This competion also helps Many CNN architecture to improve, like VGGNet, GoogleNet (Inception).

 It motivates researchers worldwide to experiment with large-scale visual recognition tasks, don't need to collect extensive datasets on their own. 

