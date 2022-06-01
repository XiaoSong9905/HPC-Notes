# Parallel ML Training

> Reference 
>
> 1. UC Berkeley CS267 Lecture 15
> 2. UC Berkeley CS294-162 Lecture 4, 6



### Overview

#### Why parallel 

模型很大，memory放不下，需要使用多个GPU/Machine



#### Type of parallell

1. data parallel
   1. batch parallel (99% of data parallel)
   2. domain parallel 
2. model parallel
3. pipeline parallel



#### Efficent training 

总的目的并不是便利总data话费的时间，而是每单位时间内学习到的东西。有些时候当batch size大了以后，反而学到的东西变小。

<img src="Note.assets/Screen Shot 2022-05-17 at 11.52.57 PM.png" alt="Screen Shot 2022-05-17 at 11.52.57 PM" style="zoom:50%;" />