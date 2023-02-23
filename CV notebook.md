

运行前先activate conda 环境

`jupyter notebook --ip 外网ip地址` 然后就可以查看服务器上的ipynb 文件了. 

vscode 双击就可以折叠cell. 还可以用outline

## Assignment1

\# TODO: how does the initialization changes the result

\# TODO: how should we choose the learning rate and the iterations

\# TODO: what happens when we initialize at 0 

When start point =-4.00, the loss will decrease slowly. Even in 100 iteration, the loss will not converage.  And the figure shows it is not reach the minimum value.

When start point =-1.00, the loss will decrease very quickly. In about 10 iteration, the loss will converage.  

When start point = 0.00,  we cannot find anything. Because in this point, the derivative(the x.grad) is nan.

When start point = 1.00,  the loss will decrease very quickly. Because in this point, the derivative(the x.grad) is nan.

When start point = 4.00,  the loss will decrease slowly. Because in this point, the derivative(the x.grad) is nan.

When learning rate = 0.1

| start point | loss                                | converage iteration | find min? | reason                                   |
| ----------- | ----------------------------------- | ------------------- | --------- | ---------------------------------------- |
| -4          | decrease slowly                     | >100                | x         | Iteration num is not enough              |
| -1          | decrease very quickly               | ~10                 | √         | close to minimum                         |
| 0           | nan                                 | nan                 | x         | the derivative at 0 (the x.grad) is nan. |
| 1           | the loss will decrease very quickly | ~10                 | x         | local minima                             |
| 4           | the loss will decrease slowly       | >100                | x         | local minima                             |

When we increase learning rate = 0.2

| start point | loss                                | converage iteration | find min? | reason                                   |
| ----------- | ----------------------------------- | ------------------- | --------- | ---------------------------------------- |
| -4          | decrease slowly                     | 80                  | √         | Iteration num is enough                  |
| -1          | decrease very quickly               | ~5                  | √         | close to minimum                         |
| 0           | nan                                 | nan                 | x         | the derivative at 0 (the x.grad) is nan. |
| 1           | the loss will decrease very quickly | ~5                  | x         | local minima                             |
| 4           | the loss will decrease slowly       | >100                | x         | local minima                             |

The learning rate controls how much the model's parameters are updated during each iteration. If the learning rate is too high, the model may overshoot the optimal parameters and fail to converge. If the learning rate is too low, the model may converge too slowly or get stuck in a local minimum. 

We can use a small learning rate, such as 0.01 or 0.001, and adjust it as needed during training. 

Iterations: The number of iterations, or epochs, determines how many times the algorithm will loop through the entire dataset during training. If the number of iterations is too low, the model may not converge to the optimal parameters. If the number of iterations is too high, the model may overfit to the training data and perform poorly on new, unseen data. A common approach is to monitor the performance of the model on a validation set during training and stop training when the performance stops improving or starts to worsen.



#### 正则化

\# learn more about regularization: https://stats.stackexchange.com/a/18765

\# tricks like (dropout, weight decay, learning rate scheduling) might help

\# see what others have tried https://stats.stackexchange.com/q/376312

To add regularization to a PyTorch neural network model, you can use various techniques, including L1 or L2 regularization, dropout, early stopping, and others.

#### weight decay



#### pooling

你就可以理解为卷积核每空两格做一次卷积，卷积核的大小是2x2， 但是卷积核的作用是取这个核里面最大的值（即特征最明显的值），而不是做卷积运算

减小计算量, 学习特征. 



多加一层    nn.Conv2d(256, 256, kernel_size=3, padding=1),

​    nn.ReLU(inplace=True),好像效果不大. 从99.26到99.28

​        \# read about http://wikipedia.org/wiki/Data_augmentation

​        \# take a look here https://pytorch.org/vision/stable/transforms.html

​        \# add some randomaization to improve the generalization

​        \# at least use 3 other transforms



#### 数据增强

notebook修改了之后要找到第一次定义的cell 重新往下运行, 不能从中间运行. 不然会很奇怪. 
