
## Scheduling & Distribution

> 参考
> 
> 1. CMU 15.418 l6

* 难点
1. want all processor working all the time
2. but want low cost solution for achieveing this balance
   1. minimize computational/schedulig overhaed 
   2. Minimize synchonization cost




### static assignment

是什么：assignment of work to thread is pre-determined. 并不一定是compile time，也可能是run time根据数据输入输出决定。

要去：able to predict cost

优点：zero runtime overhead

缺点：很那predict work，容易导致work不平均




### semi-static assignment

是什么：cost of work is predictable for nea-term future。application是不是的重新进行估计，重新进行adjustment。

常用于：particle simulation，宇宙行星关系的smilation




### dynamic assignment

* single work queue

是什么：distribution是在runtime决定的

常见构造：

<img src="Note.assets/Screen Shot 2021-10-17 at 10.49.54 AM.png" alt="Screen Shot 2021-10-17 at 10.49.54 AM" style="zoom:30%;" />

缺点：如果task被切分的比较小的话，会有较高的synchonization overhead（lock等），所以一般需要选择一个合适的task切分力度coarse granularity partitioning

* smater task scheduling 

prioritixe high cost task. 这样能避免long task在后期被取到，导致工作分配不均匀。

<img src="Note.assets/Screen Shot 2021-10-17 at 10.52.24 AM.png" alt="Screen Shot 2021-10-17 at 10.52.24 AM" style="zoom:30%;" />

<img src="Note.assets/Screen Shot 2021-10-17 at 10.52.38 AM.png" alt="Screen Shot 2021-10-17 at 10.52.38 AM" style="zoom:30%;" />

* distributed set of queues

减少了worker之间在同一个work queue上进行synchronize的cost。每一个worker现在都有自己的work queue。

当一个worker的活干完了以后，可以steal其余worker的活。cilk就是通过这种方式实现的。

<img src="Note.assets/Screen Shot 2021-10-17 at 10.54.29 AM.png" alt="Screen Shot 2021-10-17 at 10.54.29 AM" style="zoom:30%;" />

缺点：Costly synchronization/communication occurs during stealing. 

优点：increase locality. threads work on tasks they create.

steal work的时候，最好从其余的worker queue的top来steal，因为（1）divide and conquer的时候最多的工作在top (2) 最上面的数据是最老的数据，在另一个worker上没有locality

add work的时候，从work queue的bottom加入。因为这个task会一会最先被取出，同时这些数据有locality

handle work的时候，从work queue的bottom取出。为了处理最近的工作

* dependency between task

只有全部的dependency都完成的时候，才会运行对应的task。一般是由scheduler manage的

<img src="Note.assets/Screen Shot 2021-10-17 at 10.59.23 AM.png" alt="Screen Shot 2021-10-17 at 10.59.23 AM" style="zoom:33%;" />

