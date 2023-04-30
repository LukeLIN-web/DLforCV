# DLforCV

1. Read the blog post titled "Illustrated Guide to LSTM’s and GRU’s: A step by step explanation", ((https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-stepbystep-explanation-44e9eb85bf21). This blog can give you a general introduction to LSTM and GRU.

2. You can either read the original paper (https://papers.nips.cc/paper/2017/file/3f5 ee243547dee91fbd053c1c4a845aa-Paper.pdf), or read the annotated paper by Harvard’s NLP group (http://nlp.seas.harvard.edu/2018/04/03/attention.html), or the blog post “The illustrated Transformer” by Jay Alammar (http://jalammar.github.io/illustratedtransformer/).

Briefly discuss the contributions of LSTM and Transformer. Compare RNN, LSTM, and Transformer, and discuss whether they can handle extremely long sequences, and if so, how can they achieve this.

## Contribution

###  LSTM

1.  Introduce the concept of memory cells, which are used to store and update information over time. So it can keep essential words. 
2. Depending on the current input, selectively update or forget information from the previous time step.
3. Avoid gradient exploration and vanish. 

###  Transformer

1. Replacing RNNs with self-attention, designed and implemented Transformer models, the first sequence transduction model based entirely on attention.
2. Proposed scaled dot-product attention, multi-head attention, and parameter-free position representation. 
3. It inspired Bert and GPT. It did a good performance in images, audio, and videos.

## Compare

### RNN

1. The neurons in the hidden layer are connected through a hidden state.
2. Problem: The inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths as memory constraints limit batching. Transformers solve this problem.

###  LSTM

1.  Introduce the concept of forgetting cells. Make extremely long sequences possible.

###  Transformer

1. Transformer is the first transduction model based on the attention mechanism, dispensing with recurrence and convolutions entirely.
2.  Allows for significantly more parallelization, Training faster.
3. Better at handling long sequences than other sequence models like RNNs and LSTMs.

### Extremely long sequences

RNN cannot process extremely long sequences.

LSTM can process highly long sequences because it has forget gates to control the flow of information into and out of the memory cells. LSTMs can remeber important old information by selectively retaining or discarding information. 

The Transformer can process extremely long sequences. 

1. Self-attention mechanism, which allows the model to selectively attend to different parts of the input sequence when computing its output. This allows the Transformer to capture long-range dependencies between other sequence components without needing recurrent connections that can be computationally expensive to train. 
2. Multi-head attention allows the model to attend to different parts of the input sequence simultaneously and to learn multiple representations of the same sequence at different levels of abstraction.



# DLforCV

1. 阅读题为 "LSTM和GRU的图解指南 "的博文： 一步一步的解释"，（（https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-stepbystep-explanation-44e9eb85bf21）。这篇博客可以给你一个关于LSTM和GRU的总体介绍。

2. 你可以阅读原始论文（https://papers.nips.cc/paper/2017/file/3f5 ee243547dee91fbd053c1c4a845aa-Paper.pdf），或者阅读哈佛大学NLP小组的注释论文（http://nlp.seas.harvard.edu/2018/04/03/attention.html），或者Jay Alammar的博文《图解的变形器》（http://jalammar.github.io/illustratedtransformer/）。

简要讨论LSTM和Transformer的贡献。比较RNN、LSTM和Transformer，并讨论它们是否能处理极长的序列，如果能，它们如何实现。

## 贡献

### LSTM

1.  介绍记忆单元的概念，它是用来存储和更新信息的，随着时间的推移。所以它可以保留基本的词。
2.  根据当前的输入，有选择地更新或遗忘前一个时间步骤的信息。
3.  避免梯度探索和消失。

### Transformer

1. 用自我注意代替RNN，设计并实现了Transformer模型，这是第一个完全基于注意的序列转换模型。
2. 提出了缩放点积注意、多头注意和无参数位置表示。
3. 它启发了Bert和GPT。它在图像、音频和视频方面做了很好的表现。

## 比较

### RNN

1. 隐藏层中的神经元通过隐藏状态连接。
2. 问题：固有的顺序性排除了训练实例内的并行化，这在较长的序列长度上变得很关键，因为内存限制了批量化。变换器解决了这个问题。

### LSTM

1.  引入遗忘单元的概念。使极长的序列成为可能。

### Transformer

1. Transformer是第一个基于注意机制的转导模型，完全省去了递归和卷积。
2. 允许明显更多的并行化，训练速度更快。
3. 与其他序列模型如RNNs和LSTMs相比，更善于处理长序列。

### 极长的序列

RNN不能处理极长的序列。

LSTM可以处理极长的序列，因为它有遗忘门来控制信息流入和流出存储单元。LSTM可以通过选择性地保留或丢弃信息来记住重要的旧信息。

变换器可以处理极长的序列。

1. 自我注意机制，它允许模型在计算其输出时有选择地注意输入序列的不同部分。这使得Transformer能够捕捉到其他序列成分之间的长距离依赖关系，而不需要训练计算成本很高的递归连接。
2. 多头关注允许模型关注输入的不同部分。
