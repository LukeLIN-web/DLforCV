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



